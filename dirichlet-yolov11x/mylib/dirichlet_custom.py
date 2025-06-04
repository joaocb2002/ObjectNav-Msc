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
    "dirichlet_mle",
    "dirichlet_map",
    "NotConvergingError"
]

class NotConvergingError(Exception):
    pass


def logposterior(D, alpha, a_prior=2.0, b_prior=2.0):
    """
    Compute log posterior = log p(D|alpha) + log p(alpha)
    Gamma(a, b) prior over each alpha_k
    """
    N, K = D.shape
    logp = log(D).mean(axis=0)
    log_likelihood = N * (gammaln(alpha.sum()) - gammaln(alpha).sum() + ((alpha - 1) * logp).sum())
    log_prior = ((a_prior - 1) * np.log(alpha) - b_prior * alpha).sum()
    return log_likelihood + log_prior

def loglikelihood(D, alpha):
    """
    Log-likelihood of Dirichlet given data and parameters
    """
    N, K = D.shape
    logp = log(D).mean(axis=0)
    return N * (gammaln(alpha.sum()) - gammaln(alpha).sum() + ((alpha - 1) * logp).sum())

def _init_a(D):
    """Moment-based initializer from Minka (Equation 21)"""
    E = D.mean(axis=0)
    E2 = (D ** 2).mean(axis=0)
    s = (E[0] - E2[0]) / (E2[0] - E[0] ** 2)
    return s * E

def dirichlet_mle(D, tol=1e-7, maxiter=10000):
    """
    MLE estimation using Newton-style update (same as MAP but without prior).

    Parameters
    ----------
    D : (N, K) array
        Observed probability vectors
    tol : float
        Convergence threshold on log-likelihood
    maxiter : int
        Maximum number of iterations

    Returns
    -------
    alpha : (K,) array
        MLE-estimated Dirichlet parameters
    """
    N, K = D.shape
    logp = log(D).mean(axis=0)
    alpha = _init_a(D)

    for i in range(maxiter):
        alpha_sum = alpha.sum()

        # Gradient of log-likelihood only
        grad = N * (psi(alpha_sum) - psi(alpha) + logp)

        # Approximate Hessian terms
        q = -N * polygamma(1, alpha)
        z = N * polygamma(1, alpha_sum)

        # Sherman-Morrison trick
        b = (grad / q).sum() / (1/z + (1/q).sum())
        delta = (grad - b) / q
        alpha_new = alpha - delta
        alpha_new = np.maximum(alpha_new, 1e-10)

        if np.abs(loglikelihood(D, alpha_new) - loglikelihood(D, alpha)) < tol:
            return alpha_new

        alpha = alpha_new

    raise NotConvergingError("MLE estimation did not converge.")

def dirichlet_map(D, a_prior=2.0, b_prior=2.0, tol=1e-7, maxiter=10000):
    """
    MAP estimation of Dirichlet parameters via modified Newton iteration.
    Prior: Gamma(a_prior, b_prior) on each alpha_k

    Parameters
    ----------
    D : (N, K) array
        Each row is a probability vector
    a_prior : float
        Prior shape (a) — must be > 1 for concavity
    b_prior : float
        Prior rate (b)
    tol : float
        Convergence threshold on change in log-posterior
    maxiter : int
        Max number of iterations

    Returns
    -------
    (K,) array
        MAP estimate of alpha parameters
    """
    N, K = D.shape
    logp = log(D).mean(axis=0)
    alpha = _init_a(D)

    for i in range(maxiter):
        alpha_sum = alpha.sum()

        # Gradient of log-likelihood
        grad_lik = N * (psi(alpha_sum) - psi(alpha) + logp)

        # Gradient of log-prior
        grad_prior = (a_prior - 1) / alpha - b_prior

        # Total gradient
        grad = grad_lik + grad_prior

        # Approximate Hessian: diagonal and rank-one correction
        q = -N * polygamma(1, alpha)
        z = N * polygamma(1, alpha_sum)

        b = (grad / q).sum() / (1/z + (1/q).sum())
        delta = (grad - b) / q
        alpha_new = alpha - delta

        # Make sure alpha_new is positive
        if np.any(alpha_new <= 0):
            alpha_new = np.maximum(alpha_new, 1e-10)

        # Check convergence
        if np.abs(logposterior(D, alpha_new, a_prior, b_prior) - logposterior(D, alpha, a_prior, b_prior)) < tol:
            return alpha_new

        # Update alpha
        alpha = alpha_new

    raise NotConvergingError("MAP estimation did not converge.")
