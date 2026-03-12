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


def wald_logpdf(x, mu, lam):
    """
    Log PDF of the inverse Gaussian (Wald) distribution.

    Parameterization: mean mu, shape (precision) lam.
    PDF: sqrt(lam/(2*pi*x^3)) * exp(-lam*(x-mu)^2 / (2*mu^2*x)), x > 0.

    Used for DDM first-passage time: decision time T ~ Wald(mu=a/v, lam=a^2)
    when diffusion coefficient s=1 (drift v, boundary a).
    """
    x = np.asarray(x, dtype=float)
    if np.any(x <= 0):
        out = np.full_like(x, -np.inf)
        out[x > 0] = _wald_logpdf_scalar(x[x > 0], mu, lam)
        return out
    return _wald_logpdf_scalar(x, mu, lam)


def _wald_logpdf_scalar(x, mu, lam):
    """Log PDF of Wald for x > 0."""
    return (
        0.5 * np.log(lam)
        - 0.5 * np.log(2.0 * np.pi)
        - 1.5 * np.log(x)
        - lam * (x - mu) ** 2 / (2.0 * mu ** 2 * np.clip(x, 1e-300, None))
    )
