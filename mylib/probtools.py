import numpy as np

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
