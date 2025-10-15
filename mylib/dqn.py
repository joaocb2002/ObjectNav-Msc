import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional

# =========================
# Observation builder (optional helper)
# =========================

def dirichlet_mean(alpha: np.ndarray) -> np.ndarray:
    # alpha: (H, W, K)
    s = np.sum(alpha, axis=-1, keepdims=True) + 1e-8
    return alpha / s

def categorical_entropy(p: np.ndarray) -> np.ndarray:
    # p: (..., K)
    ps = np.clip(p, 1e-12, 1.0)
    return -np.sum(ps * np.log(ps), axis=-1)

def build_fullmap_obs(
    alpha: np.ndarray,         # (H,W,K) Dirichlet parameters (can be priors in free space)
    target_idx: int,
    occupancy: np.ndarray,     # (H,W)  0 = free, 1 = occupied
    agent_rc: Tuple[int, int], # (r,c)
    k_classes: int
) -> np.ndarray:
    """
    Returns (C,H,W) float32 with C=4:
      0: p_target on occupied cells (0 on free)
      1: entropy (normalized) on occupied cells (0 on free)
      2: grid_map (0 on free, 1 on occupied)
      3: agent_pos (one-hot)
    """
    H, W, K = alpha.shape

    # Dirichlet mean and entropy per cell over classes
    p_all = dirichlet_mean(alpha)                      # (H,W,K)
    p_target = p_all[..., target_idx]                  # (H,W)

    # normalized categorical entropy over K classes
    H_cat = categorical_entropy(p_all) / (math.log(k_classes + 1e-12))   # (H,W)

    # Masks
    occ = occupancy.astype(np.float32)                 # 1 on occupied, 0 on free

    # Zero-out prob/entropy on free space
    p_target_occ = p_target.astype(np.float32) * occ
    entropy_occ  = H_cat.astype(np.float32) * occ

    # Agent position map
    agent_pos = np.zeros((H, W), dtype=np.float32)
    r, c = agent_rc
    r = max(0, min(H-1, int(r))); c = max(0, min(W-1, int(c)))
    agent_pos[r, c] = 1.0

    obs = np.stack(
        [p_target_occ, entropy_occ, occ, agent_pos],
        axis=0
    )

    # Safety
    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    return obs.astype(np.float32)


# ========================= #
# DQN Network, Replay Buffer, Training
# ========================= #

class ObjectSearchQNetwork(nn.Module):
    """
    DQN for object search.
    Input:
        obs: Tensor of shape (B, C, 19, 32)
            where C = 4 (p_target_occ, entropy_occ, occupancy, agent_pos)
    Output:
        Q-values of shape (B, num_actions)
    """
    def __init__(self, in_channels: int = 4, num_actions: int = 5, feature_dim: int = 256):
        super().__init__()

        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),  # -> (32, 10, 16) for (19,32) input
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # -> (64, 5, 8)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()   # -> (64, 5, 8)
        )

        # Fully-connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 8, feature_dim),  # 64*5*8 = 2560
            nn.ReLU(),
            nn.Linear(feature_dim, num_actions)
        )

        # Orthogonal initialization (stable)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu') if m is not self.fc[-1] else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            obs (torch.Tensor): (B, C, 19, 32)
        Returns:
            torch.Tensor: Q-values (B, num_actions)
        """
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)
        q = self.fc(x)
        return q


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, obs, action, reward, next_obs, done, mask, next_mask):
        # mask and next_mask are shape (A,) arrays (0/1)
        self.buffer.append((obs, action, reward, next_obs, done, mask, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, masks, next_masks = zip(*batch)

        actions    = torch.as_tensor(actions, dtype=torch.long,    device=self.device)
        rewards    = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones      = torch.as_tensor(dones,   dtype=torch.float32, device=self.device)
        obs        = torch.as_tensor(np.stack(obs, axis=0),        dtype=torch.float32, device=self.device)
        next_obs   = torch.as_tensor(np.stack(next_obs, axis=0),   dtype=torch.float32, device=self.device)
        masks      = torch.as_tensor(np.stack(masks, axis=0),      dtype=torch.float32, device=self.device)      # (B,A)
        next_masks = torch.as_tensor(np.stack(next_masks, axis=0), dtype=torch.float32, device=self.device)       # (B,A)

        return obs, actions, rewards, next_obs, dones, masks, next_masks
    
    def __len__(self):
        return len(self.buffer)


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    q:    (B, A)
    mask: (B, A) with 1 for valid actions, 0 for invalid
    returns: (B,) argmax over valid actions
    """
    # set invalid Qs to -inf
    invalid = (mask <= 0)
    q = q.clone()
    q[invalid] = float('-inf')
    return q.argmax(dim=1)

def select_action(q_net: ObjectSearchQNetwork,
                  obs_np: np.ndarray,          # (C,H,W)
                  valid_mask_np: np.ndarray,   # (A,), 1=valid, 0=invalid
                  epsilon: float) -> int:
    A = valid_mask_np.shape[0]
    valid_idxs = np.flatnonzero(valid_mask_np > 0)

    # Safety: if mask is empty, fallback to all actions valid
    if len(valid_idxs) == 0:
        valid_idxs = np.arange(A)

    if random.random() < epsilon:
        return int(np.random.choice(valid_idxs))

    with torch.no_grad():
        device = next(q_net.parameters()).device
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)         # (1,C,H,W)
        q = q_net(obs_t)                                                 # (1,A)
        mask_t = torch.from_numpy(valid_mask_np).to(device).view(1, -1)  # (1,A)
        a = masked_argmax(q, mask_t)[0].item()
        return int(a)

def dqn_update(q_net: ObjectSearchQNetwork,
               target_net: ObjectSearchQNetwork,
               optimizer: torch.optim.Optimizer,
               replay: ReplayBuffer,
               batch_size: int,
               gamma: float):
    if len(replay) < batch_size:
        return None

    obs, actions, rewards, next_obs, dones, masks, next_masks = replay.sample(batch_size)

    # Q(s, a)
    q = q_net(obs)                                  # (B, A)
    q_sa = q.gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        # Online net chooses a* at s' over valid next actions
        q_next_online = q_net(next_obs)             # (B, A)
        invalid_next  = (next_masks <= 0)
        q_next_online_masked = q_next_online.clone()
        q_next_online_masked[invalid_next] = float('-inf')
        next_actions = q_next_online_masked.argmax(dim=1)  # (B,)

        # Target net evaluates s', a*
        q_next_target = target_net(next_obs)        # (B, A)
        q_next = q_next_target.gather(1, next_actions.view(-1,1)).squeeze(1)

        target = rewards + (1.0 - dones) * gamma * q_next

    loss = F.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())

def hard_update_target(q_net: ObjectSearchQNetwork, target_net: ObjectSearchQNetwork):
    target_net.load_state_dict(q_net.state_dict())

