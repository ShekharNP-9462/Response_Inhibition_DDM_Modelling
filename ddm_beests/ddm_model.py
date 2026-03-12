"""
Full diffusion (DDM) model for stop-signal data.

Go RTs: shifted Wald (inverse Gaussian) — RT = Ter + D, D ~ Wald(a/v, a^2)
with drift v, boundary a, non-decision time Ter (diffusion coefficient s=1).
Stop process: ex-Gaussian SSRT; race with go DDM for stop trials.
"""

import numpy as np
import pymc as pm
from scipy.stats import invgauss

from .distributions import exgauss_logpdf, wald_logpdf


def _loglik_ddm_single_subject(
    go_df, stop_df, v, a, ter, mu_ssrt, sigma_ssrt, tau_ssrt, n_mc=500
):
    """
    Log-likelihood for single subject: DDM go process + ex-Gaussian SSRT race.

    Go: RT = Ter + D, D ~ Wald(mu=a/v, lam=a^2).
    Stop: race between T_go (Ter + Wald) and T_stop ~ exGaussian.
    """
    rts_go = go_df["rt"].to_numpy()
    # Go: shifted Wald. Decision time = RT - Ter must be > 0.
    dt_go = rts_go - ter
    valid_go = dt_go > 1e-6
    if not np.all(valid_go):
        return -np.inf
    mu_wald = a / (v + 1e-9)
    lam_wald = a ** 2
    ll_go = np.sum(wald_logpdf(dt_go, mu_wald, lam_wald))

    ssd = stop_df["ssd"].to_numpy()
    rts_stop = stop_df["rt"].to_numpy()
    resp = stop_df["response"].to_numpy()
    is_inhibit = resp == "inhibit"
    is_respond = resp == "respond"

    n_trials = stop_df.shape[0]
    rng = np.random.default_rng()

    # Sample go finishing times: Ter + Wald(a/v, a^2)
    mu_w = a / (v + 1e-9)
    scale_w = a ** 2
    d_go = invgauss.rvs(mu=mu_w, scale=scale_w, size=n_mc, random_state=rng)
    t_go_samples = ter + d_go

    stop_norm = rng.normal(loc=mu_ssrt, scale=sigma_ssrt, size=n_mc)
    stop_exp = rng.exponential(scale=tau_ssrt, size=n_mc)
    t_stop_samples = stop_norm + stop_exp

    ll_stop = np.zeros(n_trials)
    for i in range(n_trials):
        d = ssd[i]
        if is_inhibit[i]:
            cond = t_stop_samples + d < t_go_samples
            p = np.clip(np.mean(cond), 1e-12, 1.0)
            ll_stop[i] = np.log(p)
        elif is_respond[i]:
            t_obs = rts_stop[i]
            dt_obs = t_obs - ter
            if dt_obs <= 0:
                ll_stop[i] = -np.inf
            else:
                log_p_tgo = wald_logpdf(dt_obs, mu_w, lam_wald)
                cond = t_obs < t_stop_samples + d
                p_cond = np.clip(np.mean(cond), 1e-12, 1.0)
                ll_stop[i] = log_p_tgo + np.log(p_cond)
        else:
            ll_stop[i] = 0.0

    return ll_go + np.sum(ll_stop)


def build_single_subject_ddm_model(go_df, stop_df, n_mc=500):
    """
    Build PyMC model: DDM for go (drift v, boundary a, Ter) + ex-Gaussian SSRT race.
    """
    with pm.Model() as model:
        # DDM parameters (all in seconds; v and a positive)
        v = pm.HalfNormal("v", sigma=2.0)   # drift rate
        a = pm.HalfNormal("a", sigma=1.0)   # boundary separation
        ter = pm.Uniform("ter", lower=0.05, upper=0.6)  # non-decision time (seconds)

        mu_ssrt = pm.Normal("mu_ssrt", mu=0.2, sigma=0.1)
        sigma_ssrt = pm.HalfNormal("sigma_ssrt", sigma=0.1)
        tau_ssrt = pm.HalfNormal("tau_ssrt", sigma=0.1)

        def logp_fn(v_, a_, ter_, mu_ssrt_, sigma_ssrt_, tau_ssrt_):
            return _loglik_ddm_single_subject(
                go_df,
                stop_df,
                float(v_),
                float(a_),
                float(ter_),
                float(mu_ssrt_),
                float(sigma_ssrt_ + 1e-6),
                float(tau_ssrt_ + 1e-6),
                n_mc=n_mc,
            )

        pm.DensityDist(
            "likelihood",
            logp_fn,
            observed={
                "v_": v,
                "a_": a,
                "ter_": ter,
                "mu_ssrt_": mu_ssrt,
                "sigma_ssrt_": sigma_ssrt,
                "tau_ssrt_": tau_ssrt,
            },
        )
    return model
