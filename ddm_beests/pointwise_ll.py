"""
Pointwise log-likelihood for WAIC/LOO: one log p(observation | theta) per trial.
"""

import numpy as np
from scipy.stats import invgauss

from .distributions import exgauss_logpdf, wald_logpdf


def pointwise_log_lik_beests(
    go_df, stop_df, mu_go, sigma_go, tau_go, mu_ssrt, sigma_ssrt, tau_ssrt, n_mc=200, rng=None
):
    """
    Per-observation log-likelihood: [ll_go_1, ..., ll_go_N, ll_stop_1, ..., ll_stop_M].
    """
    if rng is None:
        rng = np.random.default_rng()
    eps = 1e-9
    sigma_go = max(sigma_go, eps)
    tau_go = max(tau_go, eps)
    sigma_ssrt = max(sigma_ssrt, eps)
    tau_ssrt = max(tau_ssrt, eps)

    rts_go = go_df["rt"].to_numpy()
    ll_go = exgauss_logpdf(rts_go, mu_go, sigma_go, tau_go)

    ssd = stop_df["ssd"].to_numpy()
    rts_stop = stop_df["rt"].to_numpy()
    resp = stop_df["response"].to_numpy()
    is_inhibit = resp == "inhibit"
    is_respond = resp == "respond"
    n_stop = len(ssd)

    go_norm = rng.normal(loc=mu_go, scale=sigma_go, size=n_mc)
    go_exp = rng.exponential(scale=tau_go, size=n_mc)
    t_go_samples = go_norm + go_exp
    stop_norm = rng.normal(loc=mu_ssrt, scale=sigma_ssrt, size=n_mc)
    stop_exp = rng.exponential(scale=tau_ssrt, size=n_mc)
    t_stop_samples = stop_norm + stop_exp

    ll_stop = np.zeros(n_stop)
    for i in range(n_stop):
        d = ssd[i]
        if is_inhibit[i]:
            p = np.clip(np.mean(t_stop_samples + d < t_go_samples), 1e-12, 1.0)
            ll_stop[i] = np.log(p)
        elif is_respond[i]:
            t_obs = rts_stop[i]
            log_p_tgo = exgauss_logpdf(np.atleast_1d(t_obs), mu_go, sigma_go, tau_go)[0]
            p_cond = np.clip(np.mean(t_obs < t_stop_samples + d), 1e-12, 1.0)
            ll_stop[i] = log_p_tgo + np.log(p_cond)
        else:
            ll_stop[i] = 0.0

    return np.concatenate([ll_go, ll_stop])


def pointwise_log_lik_ddm(
    go_df, stop_df, v, a, ter, mu_ssrt, sigma_ssrt, tau_ssrt, n_mc=200, rng=None
):
    """Per-observation log-likelihood for DDM (shifted Wald go + ex-Gaussian SSRT race)."""
    if rng is None:
        rng = np.random.default_rng()
    eps = 1e-9
    sigma_ssrt = max(sigma_ssrt, eps)
    tau_ssrt = max(tau_ssrt, eps)

    rts_go = go_df["rt"].to_numpy()
    dt_go = rts_go - ter
    if np.any(dt_go <= 0):
        return np.full(len(go_df) + len(stop_df), -np.inf)
    mu_w = a / (v + eps)
    lam_w = a ** 2
    ll_go = wald_logpdf(dt_go, mu_w, lam_w)

    ssd = stop_df["ssd"].to_numpy()
    rts_stop = stop_df["rt"].to_numpy()
    resp = stop_df["response"].to_numpy()
    is_inhibit = resp == "inhibit"
    is_respond = resp == "respond"
    n_stop = len(ssd)

    d_go = invgauss.rvs(mu=mu_w, scale=lam_w, size=n_mc, random_state=rng)
    t_go_samples = ter + d_go
    stop_norm = rng.normal(loc=mu_ssrt, scale=sigma_ssrt, size=n_mc)
    stop_exp = rng.exponential(scale=tau_ssrt, size=n_mc)
    t_stop_samples = stop_norm + stop_exp

    ll_stop = np.zeros(n_stop)
    for i in range(n_stop):
        d = ssd[i]
        if is_inhibit[i]:
            p = np.clip(np.mean(t_stop_samples + d < t_go_samples), 1e-12, 1.0)
            ll_stop[i] = np.log(p)
        elif is_respond[i]:
            t_obs = rts_stop[i]
            dt_obs = t_obs - ter
            if dt_obs <= 0:
                ll_stop[i] = -np.inf
            else:
                log_p_tgo = wald_logpdf(np.atleast_1d(dt_obs), mu_w, lam_w)[0]
                p_cond = np.clip(np.mean(t_obs < t_stop_samples + d), 1e-12, 1.0)
                ll_stop[i] = log_p_tgo + np.log(p_cond)
        else:
            ll_stop[i] = 0.0

    return np.concatenate([ll_go, ll_stop])
