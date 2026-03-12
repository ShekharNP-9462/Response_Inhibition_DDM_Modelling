import numpy as np
from scipy.stats import norm


def exgauss_logpdf(x, mu, sigma, tau):
    """
    Log PDF of the ex-Gaussian distribution.

    Parameters
    ----------
    x : array_like
        Data.
    mu : float
        Mean of the Gaussian component.
    sigma : float
        Std of the Gaussian component (sigma > 0).
    tau : float
        Mean of the exponential component (tau > 0).
    """
    x = np.asarray(x)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)

    # Stable formulation of ex-Gaussian PDF
    z = (x - mu) / sigma
    v = sigma / tau
    arg = (z - v) / np.sqrt(2.0)
    log_pdf = (
        -np.log(tau)
        + 0.5 * v**2
        - z / tau
        + norm.logcdf(-arg * np.sqrt(2.0))
    )
    return log_pdf


def safe_logsumexp(log_w, axis=None):
    """
    Numerically-stable logsumexp for arrays of log-weights.
    """
    log_w = np.asarray(log_w)
    m = np.max(log_w, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(log_w - m), axis=axis, keepdims=True))
    if axis is not None:
        res = np.squeeze(res, axis=axis)
    return res
