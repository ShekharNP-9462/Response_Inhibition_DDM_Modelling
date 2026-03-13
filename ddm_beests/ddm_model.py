"""
Full diffusion (DDM) model for stop-signal data.

Go RTs: shifted Wald (inverse Gaussian) — RT = Ter + D, D ~ Wald(a/v, a^2)
with drift v, boundary a, non-decision time Ter (diffusion coefficient s=1).
Stop process: ex-Gaussian SSRT; race with go DDM for stop trials.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from scipy.stats import invgauss

from .distributions import exgauss_logpdf, wald_logpdf


class DDMLogLikeOp(Op):
    """PyTensor Op that wraps the black-box DDM log-likelihood (returns scalar)."""

    def __init__(self, go_df, stop_df, n_mc=500, eps=1e-6):
        self.go_df = go_df
        self.stop_df = stop_df
        self.n_mc = n_mc
        self.eps = eps

    def make_node(self, v, a, ter, mu_ssrt, sigma_ssrt, tau_ssrt):
        v = pt.as_tensor_variable(v)
        a = pt.as_tensor_variable(a)
        ter = pt.as_tensor_variable(ter)
        mu_ssrt = pt.as_tensor_variable(mu_ssrt)
        sigma_ssrt = pt.as_tensor_variable(sigma_ssrt)
        tau_ssrt = pt.as_tensor_variable(tau_ssrt)
        inputs = [v, a, ter, mu_ssrt, sigma_ssrt, tau_ssrt]
        outputs = [pt.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        (v, a, ter, mu_ssrt, sigma_ssrt, tau_ssrt) = [
            float(np.asarray(x).ravel()[0]) for x in inputs
        ]
        logp = _loglik_ddm_single_subject(
            self.go_df,
            self.stop_df,
            v,
            a,
            ter,
            mu_ssrt,
            sigma_ssrt + self.eps,
            tau_ssrt + self.eps,
            n_mc=self.n_mc,
        )
        output_storage[0][0] = np.array(logp, dtype=np.float64)


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
        v = pm.HalfNormal("v", sigma=1.5)   # drift rate (tighter)
        a = pm.HalfNormal("a", sigma=0.8)   # boundary separation
        ter = pm.Uniform("ter", lower=0.05, upper=0.5)  # non-decision time (seconds)

        mu_ssrt = pm.Normal("mu_ssrt", mu=0.22, sigma=0.08)
        sigma_ssrt = pm.HalfNormal("sigma_ssrt", sigma=0.08)
        tau_ssrt = pm.HalfNormal("tau_ssrt", sigma=0.08)

        loglike_op = DDMLogLikeOp(go_df, stop_df, n_mc=n_mc, eps=1e-6)
        pm.Potential(
            "likelihood",
            loglike_op(v, a, ter, mu_ssrt, sigma_ssrt, tau_ssrt),
        )
    return model


def get_ddm_initvals(go_df, stop_df):
    """Data-based initial values for DDM parameters (seconds)."""
    rts = go_df["rt"].dropna().to_numpy()
    if len(rts) < 2:
        return None
    mean_rt = float(np.mean(rts))
    # Rough DDM inits: ter + a/v ≈ mean_rt; set ter~0.15, a/v~0.35
    return {
        "v": 1.2,
        "a": 0.5,
        "ter": 0.18,
        "mu_ssrt": 0.22,
        "sigma_ssrt": 0.06,
        "tau_ssrt": 0.06,
    }
