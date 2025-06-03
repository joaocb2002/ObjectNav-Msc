"""dirichlet_map.py

MAP estimation of Dirichlet distribution parameters using a Gamma prior.

This module extends the MLE logic from Minka's derivation to support MAP estimation
with a Gamma(a, b) prior over each alpha_k:

    p(alpha_k) ∝ alpha_k^{a-1} * exp(-b*alpha_k)

This introduces the log-prior:

    log p(alpha_k) = (a - 1) * log(alpha_k) - b * alpha_k
    d/dalpha_k log p(alpha_k) = (a - 1) / alpha_k - b
"""

import sys
import numpy as np
from numpy.linalg import norm
from scipy.special import gammaln, psi, polygamma
from numpy import log, exp

MAXINT = sys.maxsize

__all__ = [
    "logposterior",
    "map_dirichlet",
]

class NotConvergingError(Exception):
    pass


def logposterior(D, a, a_prior=2.0, b_prior=2.0):
    """
    Compute log posterior: log p(D | alpha) + log p(alpha), where
    p(alpha_k) ∝ Gamma(a_prior, b_prior)

    Parameters
    ----------
    D : (N, K) array
        Dataset of N observations of K-dimensional probability vectors
    a : (K,) array
        Current Dirichlet parameters
    a_prior : float
        Shape parameter of Gamma prior
    b_prior : float
        Rate parameter of Gamma prior

    Returns
    -------
    float
        Log posterior value
    """
    N, K = D.shape
    logp = log(D).mean(axis=0)
    log_likelihood = N * (gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * logp).sum())
    log_prior = ((a_prior - 1) * log(a) - b_prior * a).sum()
    return log_likelihood + log_prior


def map_dirichlet(D, a_prior=2.0, b_prior=2.0, tol=1e-7, maxiter=None):
    """
    MAP estimation of Dirichlet parameters with Gamma prior over alphas

    Parameters
    ----------
    D : (N, K) array
        Dataset of N observations of K-dimensional probability vectors
    a_prior : float
        Shape parameter of Gamma prior (must be > 1 to favor alpha_k ≈ 1)
    b_prior : float
        Rate parameter of Gamma prior
    tol : float
        Convergence tolerance based on change in log-posterior
    maxiter : int or None
        Maximum number of iterations

    Returns
    -------
    (K,) array
        MAP-estimated alpha parameters
    """
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)

    if maxiter is None:
        maxiter = MAXINT

    for i in range(maxiter):
        # Gradient of log-likelihood + log-prior
        g_lik = D.shape[0] * (psi(a0.sum()) - psi(a0) + logp)
        g_prior = (a_prior - 1) / a0 - b_prior
        g_total = g_lik + g_prior

        # Hessian approximation using trigamma functions
        q = -D.shape[0] * polygamma(1, a0)
        z = D.shape[0] * polygamma(1, a0.sum())
        b = (g_total / q).sum() / (1 / z + (1 / q).sum())
        delta = (g_total - b) / q

        a1 = a0 - delta

        if np.any(a1 <= 0):
            raise NotConvergingError(f"Invalid update: some alpha_k ≤ 0 at iter {i}: {a1}")

        if abs(logposterior(D, a1, a_prior, b_prior) - logposterior(D, a0, a_prior, b_prior)) < tol:
            return a1

        a0 = a1

    raise NotConvergingError(f"MAP estimation did not converge after {maxiter} iterations.")


def _init_a(D):
    """
    Moment-based initialization of alpha parameters (from Minka's paper, eq. 21)

    Parameters
    ----------
    D : (N, K) array
        Observed data

    Returns
    -------
    (K,) array
        Initial guess for alpha
    """
    E = D.mean(axis=0)
    E2 = (D ** 2).mean(axis=0)
    s = (E[0] - E2[0]) / (E2[0] - E[0] ** 2)
    return s * E
