import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import math
from typing import Tuple


# =========================
# Observation builder (unchanged)
# =========================

def dirichlet_mean(alpha: np.ndarray) -> np.ndarray:
    s = np.sum(alpha, axis=-1, keepdims=True) + 1e-8
    return alpha / s

def categorical_entropy(p: np.ndarray) -> np.ndarray:
    ps = np.clip(p, 1e-12, 1.0)
    return -np.sum(ps * np.log(ps), axis=-1)

def build_fullmap_obs(
    alpha: np.ndarray,
    target_idx: int,
    occupancy: np.ndarray,
    agent_rc: Tuple[int, int],
    k_classes: int
) -> np.ndarray:
    H, W, K = alpha.shape
    p_all = dirichlet_mean(alpha)
    p_target = p_all[..., target_idx]
    H_cat = categorical_entropy(p_all) / (math.log(k_classes + 1e-12))
    occ = occupancy.astype(np.float32)
    p_target_occ = p_target.astype(np.float32) * occ
    entropy_occ = H_cat.astype(np.float32) * occ

    agent_pos = np.zeros((H, W), dtype=np.float32)
    r, c = agent_rc
    r = max(0, min(H-1, int(r))); c = max(0, min(W-1, int(c)))
    agent_pos[r, c] = 1.0

    obs = np.stack([p_target_occ, entropy_occ, occ, agent_pos], axis=0)
    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    return obs.astype(np.float32)


# ========================= #
# DQN Network (45x45 output)
# ========================= #

class ObjectSearchQNetwork(nn.Module):
    """
    DQN for object search.
    Input:
        obs: Tensor (B, 4, 45, 45)
    Output:
        Q-values (B, 2025)
    """
    def __init__(self, in_channels: int = 4, num_actions: int = 45 * 45, feature_dim: int = 512):
        super().__init__()

        # Convolutional encoder (same pattern as before)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),  # -> (32, 23, 23)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),  # -> (64, 12, 12)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()   # -> (64, 12, 12)
        )

        # Fully-connected layers → output 2025 actions
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, feature_dim),  # 9216 -> 512
            nn.ReLU(),
            nn.Linear(feature_dim, num_actions)
        )

        # Orthogonal initialization
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
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)
        q = self.fc(x)
        return q  # (B, 2025)


# ========================= #
# Replay Buffer (unchanged)
# ========================= #

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, obs, action, reward, next_obs, done, mask, next_mask):
        self.buffer.append((obs, action, reward, next_obs, done, mask, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, masks, next_masks = zip(*batch)
        actions    = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards    = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones      = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        obs        = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device)
        next_obs   = torch.as_tensor(np.stack(next_obs), dtype=torch.float32, device=self.device)
        masks      = torch.as_tensor(np.stack(masks), dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack(next_masks), dtype=torch.float32, device=self.device)
        return obs, actions, rewards, next_obs, dones, masks, next_masks

    def __len__(self):
        return len(self.buffer)


# ========================= #
# Masked Action Selection
# ========================= #

def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    invalid = (mask <= 0)
    q = q.clone()
    q[invalid] = float('-inf')
    return q.argmax(dim=1)

def select_action(q_net: ObjectSearchQNetwork,
                  obs_np: np.ndarray,
                  valid_mask_np: np.ndarray,
                  epsilon: float) -> int:
    A = valid_mask_np.shape[0]
    valid_idxs = np.flatnonzero(valid_mask_np > 0)
    if len(valid_idxs) == 0:
        valid_idxs = np.arange(A)
    if random.random() < epsilon:
        return int(np.random.choice(valid_idxs))
    with torch.no_grad():
        device = next(q_net.parameters()).device
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        q = q_net(obs_t)
        mask_t = torch.from_numpy(valid_mask_np).to(device).view(1, -1)
        a = masked_argmax(q, mask_t)[0].item()
        return int(a)


# ========================= #
# Masked Double DQN Update
# ========================= #

def dqn_update(q_net, target_net, optimizer, replay, batch_size, gamma):
    if len(replay) < batch_size:
        return None

    obs, actions, rewards, next_obs, dones, masks, next_masks = replay.sample(batch_size)
    q = q_net(obs)
    q_sa = q.gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        q_next_online = q_net(next_obs)
        invalid_next = (next_masks <= 0)
        q_next_online[invalid_next] = float('-inf')
        next_actions = q_next_online.argmax(dim=1)
        q_next_target = target_net(next_obs)
        q_next = q_next_target.gather(1, next_actions.view(-1, 1)).squeeze(1)
        target = rewards + (1.0 - dones) * gamma * q_next

    loss = F.smooth_l1_loss(q_sa, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def hard_update_target(q_net: ObjectSearchQNetwork, target_net: ObjectSearchQNetwork):
    target_net.load_state_dict(q_net.state_dict())
