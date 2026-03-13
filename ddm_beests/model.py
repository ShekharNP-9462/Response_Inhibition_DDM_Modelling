import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import Apply, Op

from .distributions import exgauss_logpdf, safe_logsumexp


class BEESTSLogLikeOp(Op):
    """PyTensor Op that wraps the black-box BEESTS log-likelihood (returns scalar)."""

    def __init__(self, go_df, stop_df, n_mc=500, eps=1e-3):
        self.go_df = go_df
        self.stop_df = stop_df
        self.n_mc = n_mc
        self.eps = eps

    def make_node(self, mu_go, sigma_go, tau_go, mu_ssrt, sigma_ssrt, tau_ssrt):
        mu_go = pt.as_tensor_variable(mu_go)
        sigma_go = pt.as_tensor_variable(sigma_go)
        tau_go = pt.as_tensor_variable(tau_go)
        mu_ssrt = pt.as_tensor_variable(mu_ssrt)
        sigma_ssrt = pt.as_tensor_variable(sigma_ssrt)
        tau_ssrt = pt.as_tensor_variable(tau_ssrt)
        inputs = [mu_go, sigma_go, tau_go, mu_ssrt, sigma_ssrt, tau_ssrt]
        outputs = [pt.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        (mu_go, sigma_go, tau_go, mu_ssrt, sigma_ssrt, tau_ssrt) = [
            float(np.asarray(x).ravel()[0]) for x in inputs
        ]
        logp = _loglik_single_subject(
            self.go_df,
            self.stop_df,
            mu_go,
            sigma_go + self.eps,
            tau_go + self.eps,
            mu_ssrt,
            sigma_ssrt + self.eps,
            tau_ssrt + self.eps,
            n_mc=self.n_mc,
        )
        output_storage[0][0] = np.array(logp, dtype=np.float64)


def _loglik_single_subject(go_df, stop_df, mu_go, sigma_go, tau_go, mu_ssrt, sigma_ssrt, tau_ssrt, n_mc=500):
    """
    Approximate log-likelihood for a single subject under a go/stop race model.

    go finishing times ~ exGaussian(mu_go, sigma_go, tau_go)
    stop finishing times (SSRT) ~ exGaussian(mu_ssrt, sigma_ssrt, tau_ssrt)

    For go trials: log p(RT | go params).
    For stop trials:
      - stop-inhibit: p(T_stop + SSD < T_go)
      - stop-respond with RT: p(T_go = t & T_go < T_stop + SSD)

    We use a simple Monte Carlo approximation for the stop trials.
    """
    rts_go = go_df["rt"].to_numpy()

    # Go trial log-likelihood (ex-Gaussian directly)
    ll_go = exgauss_logpdf(rts_go, mu_go, sigma_go, tau_go)

    # Stop trials
    # Monte Carlo: draw finishing times from go and stop distributions
    ssd = stop_df["ssd"].to_numpy()
    rts_stop = stop_df["rt"].to_numpy()

    # Response coding: assume "inhibit" for successful inhibition, "respond" for failed inhibition
    resp = stop_df["response"].to_numpy()
    is_inhibit = resp == "inhibit"
    is_respond = resp == "respond"

    n_trials = stop_df.shape[0]
    # Draw MC samples for go and stop processes
    # For efficiency we share draws across trials
    # Sample ex-Gaussians by sum of normal + exponential
    rng = np.random.default_rng()

    # Go process samples
    go_norm = rng.normal(loc=mu_go, scale=sigma_go, size=n_mc)
    go_exp = rng.exponential(scale=tau_go, size=n_mc)
    t_go_samples = go_norm + go_exp  # shape (n_mc,)

    # Stop process samples (SSRT)
    stop_norm = rng.normal(loc=mu_ssrt, scale=sigma_ssrt, size=n_mc)
    stop_exp = rng.exponential(scale=tau_ssrt, size=n_mc)
    t_stop_samples = stop_norm + stop_exp  # shape (n_mc,)

    ll_stop = np.zeros(n_trials)

    for i in range(n_trials):
        d = ssd[i]
        if is_inhibit[i]:
            # P(T_stop + d < T_go)
            # Approximate by Monte Carlo
            cond = t_stop_samples + d < t_go_samples
            p = np.mean(cond)
            p = np.clip(p, 1e-12, 1.0)
            ll_stop[i] = np.log(p)
        elif is_respond[i]:
            t_obs = rts_stop[i]
            # For stop-respond, we approximate:
            # p(T_go = t_obs & T_go < T_stop + d)
            # ~ p(T_go = t_obs) * P(T_go < T_stop + d | T_go = t_obs)
            # Approximate conditional probability by Monte Carlo
            # Here we treat the event T_go ~ exGauss and just use its logpdf at t_obs.
            log_p_tgo = exgauss_logpdf(t_obs, mu_go, sigma_go, tau_go)

            # Cond prob: P(t_obs < T_stop + d)
            cond = t_obs < t_stop_samples + d
            p_cond = np.mean(cond)
            p_cond = np.clip(p_cond, 1e-12, 1.0)
            ll_stop[i] = log_p_tgo + np.log(p_cond)
        else:
            # Unknown coding, ignore by giving zero contribution
            ll_stop[i] = 0.0

    return np.sum(ll_go) + np.sum(ll_stop)


def build_single_subject_model(go_df, stop_df, n_mc=500):
    """
    Build a PyMC model for a single subject's stop-signal data.

    The model uses ex-Gaussian go RTs and ex-Gaussian SSRTs, combined via
    a race model with Monte Carlo-approximated likelihood for stop trials.
    """

    # Fix small jitter to avoid zero/negative sigma/tau
    eps = 1e-3

    with pm.Model() as model:
        # Priors for go RT distribution (ex-Gaussian)
        mu_go = pm.Normal("mu_go", mu=0.4, sigma=0.2)  # seconds
        sigma_go = pm.HalfNormal("sigma_go", sigma=0.2)
        tau_go = pm.HalfNormal("tau_go", sigma=0.2)

        # Priors for SSRT distribution (ex-Gaussian)
        mu_ssrt = pm.Normal("mu_ssrt", mu=0.2, sigma=0.1)
        sigma_ssrt = pm.HalfNormal("sigma_ssrt", sigma=0.1)
        tau_ssrt = pm.HalfNormal("tau_ssrt", sigma=0.1)

        # Black-box likelihood via PyTensor Op (no gradients → Slice sampler).
        loglike_op = BEESTSLogLikeOp(go_df, stop_df, n_mc=n_mc, eps=eps)
        pm.Potential(
            "likelihood",
            loglike_op(mu_go, sigma_go, tau_go, mu_ssrt, sigma_ssrt, tau_ssrt),
        )

    return model

