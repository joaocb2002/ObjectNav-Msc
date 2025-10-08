# ppo.py
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Geometry & observation helpers (egocentric rotate + padding)
# ============================================================

HEADING2K = {
    # your sim can map its headings to these; choose one convention and keep it
    # k = number of 90° counter-clockwise rotations to make heading "face up"
    # e.g., if heading==0 means "up", set it to 0; if heading==1 means "right", set 3, etc.
    "UP": 0,
    "RIGHT": 3,
    "DOWN": 2,
    "LEFT": 1,
}

def rotate90k(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate HxW (or CxHxW) array by k * 90° CCW."""
    k = k % 4
    if k == 0:
        return arr
    if arr.ndim == 2:
        return np.rot90(arr, k=k, axes=(0, 1)).copy()
    elif arr.ndim == 3:
        # assume (C,H,W)
        return np.rot90(arr, k=k, axes=(1, 2)).copy()
    else:
        raise ValueError("rotate90k expects 2D or 3D arrays")

def pad_to_square_30(arr: np.ndarray) -> np.ndarray:
    """Zero-pad HxW or CxHxW to (30,30) spatial size (top-left aligned)."""
    if arr.ndim == 2:
        H, W = arr.shape
        out = np.zeros((30, 30), dtype=arr.dtype)
        out[:H, :W] = arr
        return out
    elif arr.ndim == 3:
        C, H, W = arr.shape
        out = np.zeros((C, 30, 30), dtype=arr.dtype)
        out[:, :H, :W] = arr
        return out
    else:
        raise ValueError("pad_to_square_30 expects 2D or 3D arrays")

def build_obs_egocentric(
    occupancy: np.ndarray,   # (H,W) 0 free, 1 obstacle
    prob_map: np.ndarray,    # (H,W) in [0,1]
    entropy_map: np.ndarray, # (H,W) real
    agent_rc: Tuple[int, int],
    heading_k: int,          # 0,1,2,3 (0 = faces up, 1 = left, 2 = down, 3 = right), or adapt to your sim
    standardize_entropy: bool = True
) -> np.ndarray:
    """
    Returns (C=4, 30, 30) float32:
      [0] occupancy (0 free, 1 obstacle)
      [1] prob_map
      [2] entropy_map (standardized per frame if standardize_entropy=True)
      [3] self-marker (one-hot)
    All channels are rotated so the agent faces 'up' (north), then zero-padded to (30,30).
    """
    H, W = occupancy.shape
    r, c = int(agent_rc[0]), int(agent_rc[1])

    occ = occupancy.astype(np.float32)
    prob = np.clip(prob_map.astype(np.float32), 0.0, 1.0)
    ent = entropy_map.astype(np.float32)

    # optional per-frame standardization for the entropy channel
    if standardize_entropy:
        mu = float(ent.mean())
        sd = float(ent.std()) + 1e-8
        ent = (ent - mu) / sd

    self_plane = np.zeros_like(occ, dtype=np.float32)
    rr = max(0, min(H - 1, r))
    cc = max(0, min(W - 1, c))
    self_plane[rr, cc] = 1.0

    stacked = np.stack([occ, prob, ent, self_plane], axis=0)  # (4,H,W)

    # Rotate by k*90° CCW so the agent's current heading faces "up"
    rot = rotate90k(stacked, k=heading_k)

    # Pad to 30x30
    rot = pad_to_square_30(rot)

    # Safety: remove NaNs/Infs
    rot = np.nan_to_num(rot, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    return rot

def forward_blocked_world(
    occupancy: np.ndarray, agent_rc: Tuple[int,int], heading_k: int
) -> bool:
    """
    Check if the "forward" cell is blocked in world frame, given heading_k (0=up,1=left,2=down,3=right).
    We use heading_k definition consistent with rotate90k (CCW).
    """
    H, W = occupancy.shape
    r, c = agent_rc
    drdc = {0: (-1, 0), 1: (0, -1), 2: (1, 0), 3: (0, 1)}  # up, left, down, right
    dr, dc = drdc[heading_k % 4]
    nr, nc = r + dr, c + dc
    if nr < 0 or nr >= H or nc < 0 or nc >= W:
        return True
    return occupancy[nr, nc] == 1

def legal_action_mask(
    occupancy: np.ndarray, agent_rc: Tuple[int,int], heading_k: int
) -> np.ndarray:
    """
    3 actions: [Forward, TurnLeft, TurnRight]
    Forward is illegal if wall/out-of-bounds in front; turns always legal.
    Returns boolean mask shape (3,) where True = legal.
    """
    f_ok = not forward_blocked_world(occupancy, agent_rc, heading_k)
    return np.array([f_ok, True, True], dtype=bool)


# =================================
# Masked categorical (for PPO + AE)
# =================================

class MaskedCategorical(torch.distributions.Categorical):
    """
    Categorical with per-sample action masks.
    mask: (B, A) boolean; invalid (False) actions get prob=0 and logprob=-inf.
    """
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            # set invalid logits to -inf
            neg_inf = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~mask, neg_inf)
        else:
            masked_logits = logits
        super().__init__(logits=masked_logits)
        self._mask = mask

    def entropy(self):
        # entropy only over valid actions
        if self._mask is None:
            return super().entropy()
        probs = self.probs
        # avoid NaNs due to 0*log(0)
        ent = -(probs * (probs.clamp_min(1e-12).log())).sum(-1)
        return ent


# =========================
# Actor-Critic Network
# =========================

class BeliefMapActorCritic(nn.Module):
    """
    Input: obs (B, 4, 30, 30)  [occupancy, prob, entropy, self]
    Output: policy logits (B, 3), state value (B, 1)
    Small CNN with Global Average Pooling (shape-agnostic once padded).
    """
    def __init__(self, in_channels: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v  = nn.Linear(hidden, 1)

        # orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01); nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight,  gain=1.00); nn.init.zeros_(self.v.bias)

    def forward(self, obs: torch.Tensor):
        x = self.encoder(obs)                # (B,64,H,W)
        x = self.gap(x).squeeze(-1).squeeze(-1)  # (B,64)
        x = self.fc(x)                       # (B,hidden)
        logits = self.pi(x)                  # (B,3)
        value  = self.v(x)                   # (B,1)
        return logits, value


# =========================
# Rollout Buffer for PPO
# =========================

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    lr: float = 2.5e-4
    ent_coef: float = 0.015
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    value_clip_range: float = 0.2

class RolloutBuffer:
    """
    Stores one PPO update's worth of transitions.
    Also stores per-step legal action masks for consistent re-evaluation.
    """
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], n_actions: int, device: str):
        C, H, W = obs_shape
        self.device = device
        self.obs      = torch.zeros((capacity, C, H, W), dtype=torch.float32, device=device)
        self.actions  = torch.zeros((capacity,), dtype=torch.long, device=device)
        self.rewards  = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.dones    = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.values   = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.masks    = torch.zeros((capacity, n_actions), dtype=torch.bool, device=device)  # legal action mask
        self.advantages = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.returns    = torch.zeros((capacity,), dtype=torch.float32, device=device)
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, done, value, logprob, legal_mask: np.ndarray):
        self.obs[self.ptr].copy_(obs)
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.values[self.ptr]   = value
        self.logprobs[self.ptr] = logprob
        self.masks[self.ptr].copy_(torch.from_numpy(legal_mask).to(self.device))
        self.ptr += 1
        if self.ptr >= self.obs.shape[0]:
            self.full = True
            self.ptr = 0

    def compute_advantages(self, last_value: float, last_done: bool, gamma: float, lam: float):
        """
        Proper GAE(λ) over the stored trajectory (assumed contiguous).
        last_value is V(s_T) for bootstrap; last_done indicates if s_T is terminal.
        """
        T = self.rewards.shape[0]
        adv = 0.0
        next_value = last_value
        next_nonterminal = 0.0 if last_done else 1.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.dones[t].item()
            delta = self.rewards[t].item() + gamma * next_value * next_nonterminal - self.values[t].item()
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantages[t] = adv
            next_value = self.values[t].item()
            next_nonterminal = nonterminal
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
                self.masks[mb_idx],
            )


# =========================
# PPO Agent
# =========================

class PPOAgent:
    """
    Minimal but strong PPO for your grid search:
      - Actions: Forward(0), TurnLeft(1), TurnRight(2)
      - Action masking on Forward
      - No Stop; no RNN
    """
    def __init__(self, in_channels: int = 4, n_actions: int = 3, obs_shape=(4,30,30), cfg: Optional[PPOConfig]=None):
        self.cfg = cfg or PPOConfig()
        self.device = self.cfg.device
        self.net = BeliefMapActorCritic(in_channels, n_actions).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        C, H, W = obs_shape
        self.buffer = RolloutBuffer(capacity=4096, obs_shape=(C,H,W), n_actions=n_actions, device=self.device)

    @torch.no_grad()
    def select_action(self, obs_np: np.ndarray, legal_mask_np: np.ndarray):
        """
        obs_np: (C,H,W) float32 numpy (already rotated & padded)
        legal_mask_np: (A,) bool numpy (True = legal). We expect A=3 here.
        returns: action(int), logprob(float), value(float)
        """
        obs = torch.from_numpy(obs_np).to(self.device)
        mask = torch.from_numpy(legal_mask_np).to(self.device)
        logits, value = self.net(obs.unsqueeze(0))
        dist = MaskedCategorical(logits=logits, mask=mask.unsqueeze(0))
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def store(self, obs_np, action, reward, done, logprob, value, legal_mask_np):
        obs = torch.from_numpy(obs_np).to(self.device)
        self.buffer.add(obs, action, reward, done, value, logprob, legal_mask_np)

    def update(self):
        # normalize advantages
        adv = self.buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages.copy_(adv)

        for _ in range(self.cfg.n_epochs):
            for obs, actions, old_logp, old_values, advantages, returns, masks in self.buffer.get_minibatches(self.cfg.batch_size):
                logits, values = self.net(obs)
                dist = MaskedCategorical(logits=logits, mask=masks)
                logp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # policy loss (clipped surrogate)
                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss (with value clipping like PPO)
                values = values.squeeze(-1)
                value_pred_clipped = old_values + (values - old_values).clamp(-self.cfg.value_clip_range, self.cfg.value_clip_range)
                v_loss1 = (values - returns).pow(2)
                v_loss2 = (value_pred_clipped - returns).pow(2)
                value_loss = torch.max(v_loss1, v_loss2).mean() * 0.5

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

        # clear buffer pointer
        self.buffer.ptr = 0
        self.buffer.full = False

    def set_lr(self, new_lr: float):
        for g in self.optim.param_groups:
            g["lr"] = new_lr


# =========================
# Reward shaping (lean)
# =========================

def compute_reward(
    target_found: bool,
    collided: bool,
    new_cell: bool,
    turning_action: bool,
    entropy_prev: Optional[np.ndarray],
    entropy_new: Optional[np.ndarray],
    success_bonus: float = 1.0,     # scaled smaller than your previous 100.0; PPO likes ~O(1) magnitudes
    step_penalty: float = -0.01,
    collision_penalty: float = -0.05,
    new_cell_bonus: float = 0.01,   # tiny novelty boost (optional)
    turning_penalty: float = 0.0,   # usually leave 0; turns are often required
    info_gain_coef: float = 0.0     # set to 0 to start; later try 0.1–0.2
) -> float:
    """
    Simple, safe shaping aligned with your belief updates.
    No Stop action; success = the sim flags 'target_found' (agent on target cell by your detector).
    """
    r = 0.0
    if target_found:
        r += success_bonus
        return r

    r += step_penalty

    if new_cell:
        r += new_cell_bonus

    if collided:
        r += collision_penalty

    if turning_action:
        r += turning_penalty

    if info_gain_coef != 0.0 and entropy_prev is not None and entropy_new is not None:
        dH = float(entropy_prev.mean() - entropy_new.mean())
        dH = max(-1.0, min(1.0, dH))
        r += info_gain_coef * dH

    return r
