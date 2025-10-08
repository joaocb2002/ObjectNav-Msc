import numpy as np

def _argmax_rc(mat):
    """Return row,col of global maximum; safe if mat is all zeros."""
    idx = int(np.argmax(mat))
    H, W = mat.shape
    return idx // W, idx % W

def _shortest_path_norm(agent_rc, goal_rc, occupancy):
    """
    Return a [0,1] normalized free-space distance from agent→goal.
    occupancy: (H,W) with 0=free, 1=occupied.
    Replace with your sim pathfinder if available.
    Fallback: L1 over free; if blocked, return 1.0.
    """
    (ar, ac), (gr, gc) = agent_rc, goal_rc
    H, W = occupancy.shape
    # trivial cases
    if (ar, ac) == (gr, gc):
        return 0.0
    
    
    # TODO: plug in simtools.shortest_path_len(...) if you have it.


    # Fallback: L1; normalize by max possible manhattan in map
    manhattan = abs(ar - gr) + abs(ac - gc)
    max_manhattan = (H - 1) + (W - 1)
    if max_manhattan == 0:
        return 1.0
    return np.clip(manhattan / max_manhattan, 0.0, 1.0)

class RewardConfig:
    def __init__(self,
                 R_success=100.0,
                 c_step=0.2,
                 c_coll=2.5,
                 c_new=0.5,
                 c_idle=0.05,
                 w_p=2.0,
                 w_H=1.0,
                 w_d=1.0,
                 phi_clip=0.5):
        self.R_success = R_success
        self.c_step = c_step
        self.c_coll = c_coll
        self.c_new = c_new
        self.c_idle = c_idle
        self.w_p = w_p
        self.w_H = w_H
        self.w_d = w_d
        self.phi_clip = phi_clip

def potential_phi(p_target_occ, entropy_occ, occupancy, agent_rc, cfg: RewardConfig):
    """Compute Φ(s) in [roughly] [-something, +something]. Inputs are per-step maps."""
    occ = occupancy.astype(np.float32)          # 1=occupied, 0=free
    # p_max over occupied cells only (p_target already zero on free in your state)
    p_max = float(np.max(p_target_occ)) if p_target_occ.size > 0 else 0.0

    # mean entropy over occupied cells only
    if np.any(occ > 0.5):
        H_mean = float(entropy_occ[occ > 0.5].mean())
    else:
        H_mean = 0.0

    # distance to current hotspot (argmax of p_target)
    goal_rc = _argmax_rc(p_target_occ)
    d_norm = _shortest_path_norm(agent_rc, goal_rc, occupancy)

    # Φ(s) = + w_p * p_max  -  w_H * H_mean  -  w_d * d_norm
    phi = cfg.w_p * p_max - cfg.w_H * H_mean - cfg.w_d * d_norm
    return phi, (p_max, H_mean, d_norm, goal_rc)

def compute_reward(
    target_found: bool,
    collided: bool,
    prev_agent_rc,
    agent_rc,
    occupancy,            # (H,W) 0=free,1=occupied
    p_target_occ_prev,    # (H,W) p_target masked to occupied (0 on free)
    entropy_occ_prev,     # (H,W) entropy masked to occupied (0 on free)
    p_target_occ_new,     # (H,W)
    entropy_occ_new,      # (H,W)
    gamma: float,
    visited_set: set,     # of (r,c) free cells seen this episode
    prev_phi: float,
    cfg: RewardConfig
):
    """
    PBRS reward + basic terms.
    Returns: reward, new_phi, aux_info
    """
    # 0) Success: terminate with big reward
    if target_found:
        return cfg.R_success, prev_phi, {"success": True}

    reward = 0.0
    info = {"success": False}

    # 1) Step penalty
    reward -= cfg.c_step
    info["step_penalty"] = -cfg.c_step

    # 2) Collision penalty
    if collided:
        reward -= cfg.c_coll
        info["collision_penalty"] = -cfg.c_coll

    # 3) Exploration bonus for entering a new free cell
    if agent_rc != prev_agent_rc:
        # entered some cell; if it's free and new, reward
        r, c = agent_rc
        if 0 <= r < occupancy.shape[0] and 0 <= c < occupancy.shape[1]:
            if occupancy[r, c] < 0.5:  # free
                if (r, c) not in visited_set:
                    reward += cfg.c_new
                    visited_set.add((r, c))
                    info["explore_bonus"] = cfg.c_new
    else:
        # no translation → tiny idle/turn tax
        reward -= cfg.c_idle
        info["idle_tax"] = -cfg.c_idle

    # 4) Potential-based shaping (policy-invariant)
    # phi_prev, aux_prev = potential_phi(p_target_occ_prev, entropy_occ_prev, occupancy, prev_agent_rc, cfg)
    # phi_new,  aux_new  = potential_phi(p_target_occ_new,  entropy_occ_new,  occupancy, agent_rc,     cfg)
    # # F = γ Φ(s') − Φ(s)
    # shaping = gamma * phi_new - phi_prev
    # shaping = float(np.clip(shaping, -cfg.phi_clip, cfg.phi_clip))
    # reward += shaping
    # info["pbrs"] = shaping
    # info["phi_prev"], info["phi_new"] = phi_prev, phi_new
    # info["aux_prev"] = {"p_max": aux_prev[0], "H_mean": aux_prev[1], "d_norm": aux_prev[2], "goal_rc": aux_prev[3]}
    # info["aux_new"]  = {"p_max": aux_new[0],  "H_mean": aux_new[1],  "d_norm": aux_new[2],  "goal_rc": aux_new[3]}
    phi_new = prev_phi  # no change
    return reward, phi_new, info
