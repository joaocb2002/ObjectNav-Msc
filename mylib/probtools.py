import numpy as np
from scipy.special import gammaln, psi

# Constants
FN_RATE = 0.3283 # Learned from calibration data

def bin_index(scale, bin_vector):
    """
    Computes the bin index for a given scale and bin vector.
    """
    bin_index = -1
    for i in range(len(bin_vector)):
        if scale <= bin_vector[i] or scale > bin_vector[-1]:
            break
        bin_index += 1

    if bin_index == -1: bin_index = 0

    return bin_index

def compute_likelihood_vector(score_vec, bbox_scale, dirichlet_priors, classes_bins):
    """
    Computes the likelihood vector for a given score vector using Dirichlet priors.
    
    Parameters:
        score_vec (np.ndarray): Softmax scores from detection, shape (K,)
        bbox_scale (float): Scale of the bounding box, used to determine the bin index.
        dirichlet_priors (dict): Dictionary of Dirichlet priors for each class,
                                where keys are class names and values are lists of Dirichlet parameters for each bin.
        classes_bins (dict): Dictionary mapping class names to their corresponding bin vectors.
    
    Returns:
        np.ndarray: Likelihood vector for each class, shape (K,)
        """

    # Initialize the likelihood vector
    likelihood_vector = []

    # Number of classes
    K = len(dirichlet_priors)

    # Go through each possible indoor class 
    for key in dirichlet_priors:
        
        # Compute the correct bin using the bounding box scale
        bin_idx = bin_index(bbox_scale, classes_bins[key])

        # Fetch the Dirichlet prior for this class and bin
        alpha = dirichlet_priors[key][bin_idx]

        if alpha is None or len(alpha) == 0:
            likelihood_vector.append(0.0)
            continue

        # Compute the likelihood vector for this class, i.e., the probability of score_vec given the prior
        l_k = dirichlet_pdf(score_vec, alpha)
        # print(f"Class: {key}, Bin Index: {bin_idx}, Scale: {bbox_scale}", end = ", ")
        # print(f"Likelihood {l_k}\n")

        # Store the likelihood in the vector
        likelihood_vector.append(l_k)

    # Normalize the likelihood vector
    likelihood_vector = np.array(likelihood_vector)
    likelihood_vector /= np.sum(likelihood_vector)

    # Multiply every element by K/(K+1)
    likelihood_vector *= K / (K + 1)

    # Append a new element for the "background" class with 1/(K + 1)
    background_likelihood = 1 / (K + 1)
    likelihood_vector = np.append(likelihood_vector, background_likelihood)

    return likelihood_vector

def dirichlet_pdf(x, alpha):
    """
    Compute the Dirichlet PDF at point x for parameters alpha.

    Parameters:
        x (array-like): K-dimensional probability vector (must sum to 1).
        alpha (array-like): K-dimensional Dirichlet parameters.

    Returns:
        float: Probability density at x.
    """
    if alpha is None or len(alpha) == 0:
        raise ValueError("Dirichlet parameters are empty.")
    
    x = np.asarray(x, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)

    if x.shape != alpha.shape:
        raise ValueError("x and alpha must have the same shape.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(alpha)):
        raise ValueError("Inputs must be finite.")
    if not np.isclose(np.sum(x), 1.0, atol=1e-6):
        raise ValueError(f"Input vector x must sum to 1. Got sum: {np.sum(x)}")
    if np.any(x < 0):
        raise ValueError("All elements of x must be >= 0.")
    if np.any(alpha <= 0):
        raise ValueError("All alpha parameters must be > 0.")
    if np.any((x == 0) & (alpha < 1)):
        return 0.0  # Avoid undefined log(0) with alpha < 1

    log_B = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    log_pdf = -log_B + np.sum((alpha - 1) * np.log(np.maximum(x, 1e-20)))

    MAX_EXP = 709.78
    MAX_FLOAT = np.finfo(np.float64).max
    return np.exp(log_pdf) if log_pdf <= MAX_EXP else MAX_FLOAT

def kaplan_update(current_belief, likelihood_vec):
    """
    Kaplan fusion update for one belief vector.
    
    Parameters:
        likelihood_vec (np.ndarray): Likelihood vector for the current observation, shape (K,)
        current_belief (np.ndarray): Current belief vector, shape (K,)
    
    Returns:
        np.ndarray: Updated belief vector, same shape as prior
    """

    # Small value to prevent division/log errors
    EPS = 1e-6

    # Sum of product of likelihood and current belief element-wise
    if likelihood_vec.shape != current_belief.shape:
        raise ValueError(f"Shape mismatch: likelihood_vec shape {likelihood_vec.shape} and current_belief shape {current_belief.shape} must be the same.")
    dot_prod = np.sum(likelihood_vec * current_belief)

    # Minimum likelihood value
    min_val = np.min(likelihood_vec)

    # Compute normalization factor to avoid division errors
    normalization_factor = dot_prod + EPS

    # Compute the likelihood ratio term
    likelihood_ratio = likelihood_vec / normalization_factor

    # Compute the minimum likelihood term
    min_likelihood_term = min_val / normalization_factor

    # Apply the Kaplan update equation
    updated = current_belief * (1 + likelihood_ratio) / (1 + min_likelihood_term)

    return updated

def compute_entropy(belief_vector):
    """
    Computes the entropy of a Dirichlet belief vector.

    Parameters:
        belief_vector (np.ndarray): Current belief vector for a cell (shape: [K+1])

    Returns:
        float: Entropy of the belief vector
    """
    belief_vector = np.clip(belief_vector, 1e-6, None)  # Avoid log(0)
    total = np.sum(belief_vector)
    normalized_belief = belief_vector / total
    entropy = -np.sum(normalized_belief * np.log(normalized_belief))
    return entropy

def compute_background_likelihood_vector(distance, num_classes, alpha=0.125):
    """
    Computes the background likelihood vector for an empty cell based on distance.

    Parameters:
        distance (float): Distance to the cell.
        num_classes (int): Number of classes, default is 1.
        alpha (float): Hyperparameter for the exponential decay, default is 0.1.

    Returns:
        np.ndarray: Background likelihood vector for the empty cell.
    """
    # Initialize the background likelihood vector
    background_likelihood_vector = np.zeros(num_classes + 1)

    # Compute the likelihood based on distance
    value = 1 / (1 + alpha * distance)*(1-FN_RATE) 

    # Set the background likelihood for the empty cell
    background_likelihood_vector[-1] = value

    # Set the likelihood for the other classes
    value = (1 - value) / num_classes
    background_likelihood_vector[:-1] = value

    return background_likelihood_vector