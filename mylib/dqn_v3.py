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
    theta: float,              # radians
    k_classes: int
) -> np.ndarray:
    """
    Returns (C,H,W) float32 with C=6:
      0: p_target on occupied cells (0 on free)
      1: entropy (normalized) on occupied cells (0 on free)
      2: free_mask (0 on free, 1 on occupied)
      3: agent_pos (one-hot)
      4: cos(theta) map
      5: sin(theta) map
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
    # Safety: clamp in case of rounding
    r = max(0, min(H-1, int(r))); c = max(0, min(W-1, int(c)))
    agent_pos[r, c] = 1.0

    # Orientation maps
    cos_map = np.full((H, W), np.cos(theta), dtype=np.float32)
    sin_map = np.full((H, W), np.sin(theta), dtype=np.float32)

    obs = np.stack(
        [p_target_occ, entropy_occ, occ, agent_pos, cos_map, sin_map],
        axis=0
    )

    # # Print every channel sequentially
  
    # # Target prob
    # print("Channel 0 (Target prob): ")
    # for i in range(obs[0].shape[0]):
    #     print(f"\n")
    #     for j in range(obs[0].shape[1]):
    #         print(f"{obs[0][i][j]:.2f} ", end="")

    # # Entropy
    # print("\nChannel 1 (Entropy): ")
    # for i in range(obs[1].shape[0]):
    #     print(f"\n")
    #     for j in range(obs[1].shape[1]):
    #         print(f"{obs[1][i][j]:.2f} ", end="")

    # # Occupancy
    # print("\nChannel 2 (Occupancy): ")
    # for i in range(obs[2].shape[0]):
    #     print(f"\n")
    #     for j in range(obs[2].shape[1]):
    #         print(f"{obs[2][i][j]} ", end="")

    # # Agent pos
    # print("\nChannel 3 (Agent pos): ")
    # for i in range(obs[3].shape[0]):
    #     print(f"\n")
    #     for j in range(obs[3].shape[1]):
    #         print(f"{obs[3][i][j]} ", end="")

    # # Cosine
    # print("\nChannel 4 (Cosine): ")
    # for i in range(obs[4].shape[0]):
    #     print(f"\n")
    #     for j in range(obs[4].shape[1]):
    #         print(f"{obs[4][i][j]:.2f} ", end="")

    # # Sine
    # print("\nChannel 5 (Sine): ")
    # for i in range(obs[5].shape[0]):
    #     print(f"\n")
    #     for j in range(obs[5].shape[1]):
    #         print(f"{obs[5][i][j]:.2f} ", end="")
    # print("\n")
    
    # Safety
    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
    return obs.astype(np.float32)


# =========================
# DQN Network, Replay Buffer, Training
# =========================
class ObjectSearchQNetwork(nn.Module):
    """
    Input:  obs (B, C, 42, 36)
            where C = 6 (p_target_occ, entropy_occ, occupancy, agent_pos, cosθ, sinθ)
    Output: Q-values (B, num_actions)
    """
    def __init__(self, in_channels: int = 6, num_actions: int = 3, feature_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),   # (6,42,36) -> (32,21,18)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),          nn.ReLU(),   # -> (64,11,9)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),          nn.ReLU(),   # -> (64,11,9)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*11*9, feature_dim),  # 64*11*9 = 6336
            nn.ReLU(),
            nn.Linear(feature_dim, num_actions)
        )

        # Orthogonal init (stable)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu') if m is not self.fc[-1] else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, 42, 36)
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)  # -> (B, 6336)
        q = self.fc(x)
        return q



class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, obs, action, reward, next_obs, done):
        # store CPU tensors or numpy; we’ll convert on sample
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        # Convert to tensors (B,C,H,W)
        actions   = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards   = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones     = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_obs  = torch.as_tensor(np.stack(next_obs, axis=0), dtype=torch.float32, device=self.device)
        obs       = torch.as_tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


def select_action(q_net: ObjectSearchQNetwork, obs_np: np.ndarray, epsilon: float, num_actions: int) -> int:
    """
    obs_np: (C,38,52) numpy float32
    """
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    with torch.no_grad():
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(next(q_net.parameters()).device)  # (1,C,H,W)
        q = q_net(obs_t)  # (1,A)
        return int(q.argmax(dim=1).item())


def dqn_update(q_net: ObjectSearchQNetwork,
               target_net: ObjectSearchQNetwork,
               optimizer: torch.optim.Optimizer,
               replay: ReplayBuffer,
               batch_size: int,
               gamma: float):
    """
    Double DQN:
      a* = argmax_a Q_online(s', a)
      target = r + (1-done) * gamma * Q_target(s', a*)
    """
    if len(replay) < batch_size:
        return None

    obs, actions, rewards, next_obs, dones = replay.sample(batch_size)
    # Q(s,a)
    q = q_net(obs)                                  # (B,A)
    q_sa = q.gather(1, actions.view(-1,1)).squeeze(1)

    with torch.no_grad():
        # Online net chooses action at s'
        q_next_online = q_net(next_obs)             # (B,A)
        next_actions = q_next_online.argmax(dim=1)  # (B,)
        # Target net evaluates s', a*
        q_next_target = target_net(next_obs)        # (B,A)
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


# =========================
# Reward shaping function
# =========================

def compute_reward(
    target_found: bool,
    success_bonus: float = 10.0,
    step_penalty: float = -0.01
) -> float:
    """
    Compute DQN reward with simple shaping:
      - success bonus
      - step penalty
      - collision penalty
      - global entropy reduction
    """
    reward = 0.0

    # Success condition
    if target_found:
        reward += success_bonus
        return reward

    # Always penalize time
    reward += step_penalty

    return reward
