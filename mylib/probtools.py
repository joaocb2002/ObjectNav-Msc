import numpy as np
from scipy.special import gammaln, psi

def kaplan_update(prior, score_vec, dirichlet_alpha, background_idx=None):
    """
    Kaplan fusion update for one belief vector.
    
    Parameters:
        prior (np.ndarray): Current belief vector (Dirichlet parameters), shape (K+1,)
        score_vec (np.ndarray): Softmax scores from detection, shape (K,)
        dirichlet_alpha (np.ndarray): Dirichlet prior for this distance, shape (K,)
        background_idx (int or None): Index of background class (default is last)
    
    Returns:
        np.ndarray: Updated belief vector, same shape as prior
    """

    # Ensure shapes
    K = len(score_vec)
    if background_idx is None:
        background_idx = K  # Assume background is last class

    # Small value to prevent division/log errors
    EPS = 1e-6

    # Clamp zeros for stability
    score_vec = np.clip(score_vec, EPS, 1.0)
    score_vec = score_vec / np.sum(score_vec)

    # --- Compute log-likelihood for each class ---
    log_likelihood = []
    for k in range(K):
        alpha_k = dirichlet_alpha[k]
        # Likelihood of score_vec under Dir(alpha_k) (independent approx.)
        log_likelihood.append(alpha_k * np.log(score_vec[k] + EPS))
    log_likelihood = np.array(log_likelihood)

    # --- Convert to pseudo-likelihood vector ---
    # Exponentiate and normalize
    dirichlet_like = np.exp(log_likelihood - np.max(log_likelihood))  # numerical stability
    dirichlet_like = dirichlet_like * ((1.0 - 1.0 / (K + 1)) / np.sum(dirichlet_like))
    dirichlet_like = np.append(dirichlet_like, 1.0 / (K + 1))  # add background class

    # --- Apply Kaplan update rule ---
    beta = np.copy(prior)
    dot_prod = np.sum(dirichlet_like * beta)
    min_val = np.min(dirichlet_like)

    # Apply the update equation
    updated = beta * (1 + (dirichlet_like / (dot_prod + EPS)) / (1 + min_val / (dot_prod + EPS)))

    return updated


def get_expected_score_vector(belief_vector, epsilon=1e-6):
    """
    Converts a Dirichlet belief vector into a normalized expected score vector.

    Parameters:
        belief_vector (np.ndarray): Current belief vector for a cell (shape: [K+1])
        epsilon (float): Small value to avoid division by zero

    Returns:
        np.ndarray: Normalized expected probability vector (shape: [K+1])
    """
    belief_vector = np.clip(belief_vector, epsilon, None)
    expected_scores = belief_vector / np.sum(belief_vector)
    return expected_scores

def kl_dirichlet(p, q):
    """
    Computes the KL divergence between two Dirichlet distributions p and q.

    Parameters:
        p (np.ndarray): Dirichlet parameters (belief before)
        q (np.ndarray): Dirichlet parameters (belief after)

    Returns:
        float: KL divergence D_KL(q || p)
    """
    p = np.clip(p, 1e-6, None)
    q = np.clip(q, 1e-6, None)

    p0 = np.sum(p)
    q0 = np.sum(q)

    kl = gammaln(q0) - gammaln(p0)
    kl -= np.sum(gammaln(q)) - np.sum(gammaln(p))
    kl += np.sum((q - p) * (psi(q) - psi(q0)))

    return kl

def compute_total_kl(simulated_map, current_map):
    """
    Computes the total KL divergence between two belief maps.

    Parameters:
        simulated_map (list of list of np.ndarray): Simulated belief map (Dirichlet per cell)
        current_map (list of list of np.ndarray): Current belief map (Dirichlet per cell)

    Returns:
        float: Total KL divergence across all cells
    """
    total_kl = 0.0
    for i in range(len(current_map)):
        for j in range(len(current_map[0])):
            p = current_map[i][j]
            q = simulated_map[i][j]
            total_kl += kl_dirichlet(p, q)
    return total_kl


