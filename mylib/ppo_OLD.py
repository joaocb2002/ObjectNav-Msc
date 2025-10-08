# ppo.py
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Actor-Critic Network
# =========================

class BeliefMapActorCritic(nn.Module):
    """
    Input: obs (B, C, 19, 32)
    Output: policy logits (B, A), state value (B, 1)
    """
    def __init__(self, in_channels: int, n_actions: int, feature_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.ReLU(),   # -> (32, 10, 16)
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),            # -> (64, 5, 8)
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),            # -> (64, 5, 8)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 8, feature_dim),  # <-- changed from 64*10*15
            nn.ReLU(),
        )
        self.pi = nn.Linear(feature_dim, n_actions)
        self.v  = nn.Linear(feature_dim, 1)


        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs: torch.Tensor):
        x = self.encoder(obs)                # (B,64,10,15)
        x = x.reshape(x.size(0), -1)         # (B, 64*10*15)
        x = self.fc(x)                       # (B, feature_dim)
        logits = self.pi(x)                  # (B, A)
        value  = self.v(x)                   # (B, 1)
        return logits, value


# =========================
# Rollout Buffer for PPO
# =========================

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class RolloutBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], device: str):
        C, H, W = obs_shape
        self.device = device
        self.obs      = torch.zeros((capacity, C, H, W), dtype=torch.float32, device=device)
        self.actions  = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rewards  = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones    = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.values   = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.returns    = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done, value, logprob):
        self.obs[self.ptr].copy_(obs)
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.values[self.ptr]   = value
        self.logprobs[self.ptr] = logprob
        self.ptr += 1
        if self.ptr >= self.obs.shape[0]:
            self.full = True
            self.ptr = 0

    def compute_advantages(self, last_value: float, gamma: float, lam: float):
        """
        GAE(λ): compute advantages and returns in-place.
        Assumes the buffer is "full" batch (contiguous episode chunks or truncated).
        """
        T = self.rewards.shape[0]
        adv = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - self.dones[t].item()
            delta = self.rewards[t].item() + gamma * last_value * mask - self.values[t].item()
            adv = delta + gamma * lam * mask * adv
            self.advantages[t] = adv
            last_value = self.values[t].item()  # bootstrap recursively on value
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int):
        idxs = torch.randperm(self.obs.shape[0], device=self.device)
        for start in range(0, len(idxs), batch_size):
            mb_idx = idxs[start:start+batch_size]
            yield (
                self.obs[mb_idx],
                self.actions[mb_idx],
                self.logprobs[mb_idx],
                self.values[mb_idx],
                self.advantages[mb_idx],
                self.returns[mb_idx],
            )


# =========================
# PPO Agent
# =========================

class PPOAgent:
    def __init__(self, in_channels: int, n_actions: int, obs_shape=(6,19,32), cfg: Optional[PPOConfig]=None):
        self.cfg = cfg or PPOConfig()
        self.device = self.cfg.device
        self.net = BeliefMapActorCritic(in_channels, n_actions).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        C, H, W = obs_shape
        self.buffer = RolloutBuffer(capacity=1024, obs_shape=(C,H,W), device=self.device)

    @torch.no_grad()
    def select_action(self, obs_np: np.ndarray):
        """
        obs_np: (C,H,W) float32 numpy
        returns: action(int), logprob(float), value(float)
        """
        obs = torch.from_numpy(obs_np).to(self.device)
        logits, value = self.net(obs.unsqueeze(0))
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def store(self, obs_np, action, reward, done, logprob, value):
        obs = torch.from_numpy(obs_np).to(self.device)
        self.buffer.add(obs, action, reward, done, value, logprob)

    def update(self):
        # normalize advantages
        adv = self.buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages.copy_(adv)

        for _ in range(self.cfg.n_epochs):
            for obs, actions, old_logp, old_values, advantages, returns in self.buffer.get_minibatches(self.cfg.batch_size):
                logits, values = self.net(obs)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # policy loss (clipped surrogate)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss (clipped)
                value_clipped = old_values + torch.clamp(values.squeeze(-1) - old_values,
                                                         -0.2, 0.2)
                v_loss1 = (values.squeeze(-1) - returns) ** 2
                v_loss2 = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

        # clear buffer pointer (simple “one big batch” usage)
        self.buffer.ptr = 0
        self.buffer.full = False

    def set_lr(self, new_lr: float):
        for g in self.optim.param_groups:
            g["lr"] = new_lr


# =========================
# Reward shaping function
# =========================

def compute_reward(
    target_found: bool,
    collided: bool,
    new_cell: bool,
    turning_action: bool,
    entropy_prev: np.ndarray,
    entropy_new: np.ndarray,
    success_bonus: float = 100.0,
    new_cell_bonus: float = 1.0,
    step_penalty: float = -0.1,
    collision_penalty: float = -5.0,
    turning_penalty: float = -3.0,
    info_gain_coef: float = 0.2
) -> float:
    """
    Compute PPO reward with simple shaping:
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

    # Reward new cell discovery
    if new_cell:
        reward += new_cell_bonus

    # Penalize collision
    if collided:
        reward += collision_penalty

    # Penalize turning actions
    if turning_action:
        reward += turning_penalty

    # Reward global entropy reduction
    # if entropy_prev is not None and entropy_new is not None:
    #     mean_prev = entropy_prev.mean()
    #     mean_new = entropy_new.mean()
    #     dI = np.clip(mean_prev - mean_new, -1.0, 1.0)
    #     reward += info_gain_coef * dI

    return reward


