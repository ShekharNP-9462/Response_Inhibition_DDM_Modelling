"""
Goodness-of-fit assessment: visual (PPC, QP plot) and quantitative (WAIC, LOO, R̂).

1. PPC: Go RT and inhibition curve (observed vs posterior predictive).
2. QP plot: Observed vs predicted go RT quantiles (diagnose fast/slow tail fit).
3. WAIC / LOO: Information criteria for model comparison (lower = better).
4. R̂ (Gelman-Rubin): MCMC convergence (R̂ ≈ 1, e.g. < 1.01).
"""

import argparse
import json
import os
import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import invgauss

from ddm_beests.io import prepare_rashi_single_subject_for_model
from ddm_beests.pointwise_ll import pointwise_log_lik_beests, pointwise_log_lik_ddm


def _sample_exgauss(mu, sigma, tau, size, rng):
    n = rng.normal(loc=mu, scale=sigma, size=size)
    e = rng.exponential(scale=tau, size=size)
    return n + e


def _sample_wald_go(ter, v, a, size, rng):
    mu_w = a / (v + 1e-9)
    scale_w = a ** 2
    d = invgauss.rvs(mu=mu_w, scale=scale_w, size=size, random_state=rng)
    return ter + d


def run_ppc_beests(go_df, stop_df, idata, n_ppc_draws=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    post = idata.posterior
    n_go = len(go_df)
    ssd = stop_df["ssd"].to_numpy()
    n_stop = len(ssd)
    mu_go = post["mu_go"].values.flatten()
    sigma_go = post["sigma_go"].values.flatten()
    tau_go = post["tau_go"].values.flatten()
    mu_ssrt = post["mu_ssrt"].values.flatten()
    sigma_ssrt = post["sigma_ssrt"].values.flatten()
    tau_ssrt = post["tau_ssrt"].values.flatten()
    n_avail = len(mu_go)
    idx = rng.choice(n_avail, size=min(n_ppc_draws, n_avail), replace=False)
    go_ppc = np.zeros((len(idx), n_go))
    inhib_ppc = np.zeros((len(idx), n_stop), dtype=bool)
    for k, i in enumerate(idx):
        go_ppc[k] = _sample_exgauss(mu_go[i], sigma_go[i], tau_go[i], n_go, rng)
        t_go = _sample_exgauss(mu_go[i], sigma_go[i], tau_go[i], n_stop, rng)
        t_stop = _sample_exgauss(mu_ssrt[i], sigma_ssrt[i], tau_ssrt[i], n_stop, rng)
        inhib_ppc[k] = (t_stop + ssd) < t_go
    return go_ppc, inhib_ppc


def run_ppc_ddm(go_df, stop_df, idata, n_ppc_draws=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    post = idata.posterior
    n_go = len(go_df)
    ssd = stop_df["ssd"].to_numpy()
    n_stop = len(ssd)
    v = post["v"].values.flatten()
    a = post["a"].values.flatten()
    ter = post["ter"].values.flatten()
    mu_ssrt = post["mu_ssrt"].values.flatten()
    sigma_ssrt = post["sigma_ssrt"].values.flatten()
    tau_ssrt = post["tau_ssrt"].values.flatten()
    n_avail = len(v)
    idx = rng.choice(n_avail, size=min(n_ppc_draws, n_avail), replace=False)
    go_ppc = np.zeros((len(idx), n_go))
    inhib_ppc = np.zeros((len(idx), n_stop), dtype=bool)
    for k, i in enumerate(idx):
        go_ppc[k] = _sample_wald_go(ter[i], v[i], a[i], n_go, rng)
        t_go = _sample_wald_go(ter[i], v[i], a[i], n_stop, rng)
        t_stop = _sample_exgauss(mu_ssrt[i], sigma_ssrt[i], tau_ssrt[i], n_stop, rng)
        inhib_ppc[k] = (t_stop + ssd) < t_go
    return go_ppc, inhib_ppc


def compute_log_likelihood_idata(idata, go_df, stop_df, model_type, n_mc_ll=200, rng=None, max_draws=None):
    """Build pointwise log_likelihood (chain, draw, obs) and add to idata for WAIC/LOO."""
    if rng is None:
        rng = np.random.default_rng(42)
    post = idata.posterior
    n_obs = len(go_df) + len(stop_df)
    chains = post.coords["chain"].values
    all_draws = post.coords["draw"].values
    if max_draws is not None and len(all_draws) > max_draws:
        idx = np.linspace(0, len(all_draws) - 1, max_draws, dtype=int)
        draws = all_draws[idx]
    else:
        draws = all_draws
    n_chain = len(chains)
    n_draw = len(draws)
    log_lik = np.full((n_chain, n_draw, n_obs), np.nan)
    for c, chain in enumerate(chains):
        for d, draw in enumerate(draws):
            if model_type == "beests":
                mu_go = float(post["mu_go"].sel(chain=chain, draw=draw).values)
                sigma_go = float(post["sigma_go"].sel(chain=chain, draw=draw).values)
                tau_go = float(post["tau_go"].sel(chain=chain, draw=draw).values)
                mu_ssrt = float(post["mu_ssrt"].sel(chain=chain, draw=draw).values)
                sigma_ssrt = float(post["sigma_ssrt"].sel(chain=chain, draw=draw).values)
                tau_ssrt = float(post["tau_ssrt"].sel(chain=chain, draw=draw).values)
                log_lik[c, d, :] = pointwise_log_lik_beests(
                    go_df, stop_df, mu_go, sigma_go, tau_go, mu_ssrt, sigma_ssrt, tau_ssrt, n_mc=n_mc_ll, rng=rng
                )
            else:
                v = float(post["v"].sel(chain=chain, draw=draw).values)
                a = float(post["a"].sel(chain=chain, draw=draw).values)
                ter = float(post["ter"].sel(chain=chain, draw=draw).values)
                mu_ssrt = float(post["mu_ssrt"].sel(chain=chain, draw=draw).values)
                sigma_ssrt = float(post["sigma_ssrt"].sel(chain=chain, draw=draw).values)
                tau_ssrt = float(post["tau_ssrt"].sel(chain=chain, draw=draw).values)
                log_lik[c, d, :] = pointwise_log_lik_ddm(
                    go_df, stop_df, v, a, ter, mu_ssrt, sigma_ssrt, tau_ssrt, n_mc=n_mc_ll, rng=rng
                )
    # Add to idata: ArviZ expects log_likelihood group with dims (chain, draw, obs)
    obs_dim = np.arange(n_obs)
    ds = xr.Dataset(
        {"log_lik": (["chain", "draw", "obs"], log_lik)},
        coords={"chain": chains, "draw": draws, "obs": obs_dim},
    )
    idata_with_ll = idata.copy()
    idata_with_ll.add_groups(log_likelihood=ds)
    return idata_with_ll


def plot_ppc_and_qp(go_df, stop_df, go_ppc, inhib_ppc, model_name, subject_id, output_path):
    """Three panels: Go RT PPC, Inhibition curve PPC, QP plot (observed vs predicted go RT quantiles)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1) Go RT PPC
    ax = axes[0]
    rts_obs = go_df["rt"].dropna().to_numpy() * 1000
    ax.hist(rts_obs, bins=25, density=True, color="black", alpha=0.5, label="Observed")
    for i in range(min(30, go_ppc.shape[0])):
        ax.hist(go_ppc[i] * 1000, bins=25, density=True, histtype="step", color="steelblue", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Go RT (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Go RT: observed vs posterior predictive")
    ax.legend(loc="upper right")

    # 2) Inhibition curve PPC
    ax = axes[1]
    ssd = stop_df["ssd"].to_numpy() * 1000
    obs_inhibit = (stop_df["response"] == "inhibit").to_numpy()
    bins = np.percentile(ssd, [0, 25, 50, 75, 100])
    bins[-1] += 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    obs_prop = []
    for j in range(len(bins) - 1):
        m = (ssd >= bins[j]) & (ssd < bins[j + 1])
        obs_prop.append(obs_inhibit[m].mean() if m.sum() > 0 else np.nan)
    obs_prop = np.array(obs_prop)
    valid = ~np.isnan(obs_prop)
    ax.plot(bin_centers[valid], obs_prop[valid], "ko-", label="Observed", markersize=8)
    pred_props = np.zeros((inhib_ppc.shape[0], len(bins) - 1))
    for k in range(inhib_ppc.shape[0]):
        for j in range(len(bins) - 1):
            m = (ssd >= bins[j]) & (ssd < bins[j + 1])
            pred_props[k, j] = inhib_ppc[k][m].mean() if m.sum() > 0 else np.nan
    pred_mean = np.nanmean(pred_props, axis=0)
    pred_lo = np.nanpercentile(pred_props, 2.5, axis=0)
    pred_hi = np.nanpercentile(pred_props, 97.5, axis=0)
    ax.fill_between(bin_centers, pred_lo, pred_hi, color="steelblue", alpha=0.3)
    ax.plot(bin_centers, pred_mean, "b-", label="Predicted (94% CI)")
    ax.set_xlabel("SSD (ms)")
    ax.set_ylabel("P(inhibit)")
    ax.set_title("Inhibition curve: observed vs posterior predictive")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.05, 1.05)

    # 3) QP plot: observed vs predicted go RT quantiles
    ax = axes[2]
    q_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q_obs = np.quantile(rts_obs, q_levels)
    q_pred = np.quantile(go_ppc * 1000, q_levels, axis=1)  # (n_levels, n_draws)
    q_pred_mean = np.mean(q_pred, axis=1)
    q_pred_lo = np.percentile(q_pred, 2.5, axis=1)
    q_pred_hi = np.percentile(q_pred, 97.5, axis=1)
    ax.fill_between(q_obs, q_pred_lo, q_pred_hi, color="steelblue", alpha=0.3)
    ax.plot(q_obs, q_pred_mean, "b-o", label="Predicted (94% CI)")
    ax.plot(q_obs, q_obs, "k--", alpha=0.7, label="Perfect fit (45°)")
    ax.set_xlabel("Observed go RT quantile (ms)")
    ax.set_ylabel("Predicted go RT quantile (ms)")
    ax.set_title("QP plot: slow/fast tail fit")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    fig.suptitle(f"{subject_id} — {model_name} goodness of fit")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rhat(rhat_series, output_path):
    """Bar plot of R̂ by parameter (flag R̂ > 1.01)."""
    fig, ax = plt.subplots(figsize=(6, max(3, len(rhat_series) * 0.35)))
    params = list(rhat_series.index)
    vals = list(rhat_series.values)
    colors = ["#e74c3c" if v > 1.01 else "#2ecc71" for v in vals]
    ax.barh(params, vals, color=colors)
    ax.axvline(1.0, color="black", linestyle="-", linewidth=0.8)
    ax.axvline(1.01, color="gray", linestyle="--", label="R̂ = 1.01")
    ax.set_xlabel("R̂ (Gelman-Rubin)")
    ax.set_title("MCMC convergence: R̂ ≈ 1 is good")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Goodness-of-fit: PPC, QP plot, WAIC/LOO, R̂.")
    parser.add_argument("subject_dir", type=str)
    parser.add_argument("trace_nc", type=str)
    parser.add_argument("--model", type=str, choices=["beests", "ddm"], required=True)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--n-ppc", type=int, default=100)
    parser.add_argument("--n-waic-draws", type=int, default=None, help="Max draws for WAIC/LOO (default: all; use e.g. 300 for speed).")
    parser.add_argument("--n-mc-ll", type=int, default=200, help="MC samples per trial for pointwise log_lik.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    subject_dir = os.path.abspath(args.subject_dir)
    subject_id = os.path.basename(subject_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading trace and data...")
    idata = az.from_netcdf(args.trace_nc)
    go_df, stop_df = prepare_rashi_single_subject_for_model(subject_dir)
    model_name = "BEESTS" if args.model == "beests" else "DDM"

    # ----- PPC + QP -----
    print("Running posterior predictive simulation...")
    if args.model == "beests":
        go_ppc, inhib_ppc = run_ppc_beests(go_df, stop_df, idata, n_ppc_draws=args.n_ppc, rng=rng)
    else:
        go_ppc, inhib_ppc = run_ppc_ddm(go_df, stop_df, idata, n_ppc_draws=args.n_ppc, rng=rng)
    out_ppc = os.path.join(args.output_dir, f"{subject_id}_{args.model}_ppc.png")
    plot_ppc_and_qp(go_df, stop_df, go_ppc, inhib_ppc, model_name, subject_id, out_ppc)
    print(f"PPC + QP figure saved to {out_ppc}")

    # ----- R̂ -----
    summary = az.summary(idata, var_names=None)
    if "r_hat" in summary.columns:
        rhat = summary["r_hat"]
        out_rhat = os.path.join(args.output_dir, f"{subject_id}_{args.model}_rhat.png")
        plot_rhat(rhat, out_rhat)
        print(f"R̂ figure saved to {out_rhat}")
        rhat_max = float(rhat.max())
        rhat_ok = rhat_max <= 1.01
    else:
        rhat = None
        rhat_max = None
        rhat_ok = None

    # ----- WAIC / LOO -----
    print("Computing pointwise log-likelihood for WAIC/LOO (this may take a minute)...")
    t0 = time.perf_counter()
    idata_ll = compute_log_likelihood_idata(
        idata, go_df, stop_df, args.model, n_mc_ll=args.n_mc_ll, rng=rng, max_draws=args.n_waic_draws
    )
    elapsed = time.perf_counter() - t0
    print(f"  Pointwise LL computed in {elapsed:.1f} s")
    try:
        waic = az.waic(idata_ll, var_name="log_lik")
        loo = az.loo(idata_ll, var_name="log_lik")
        waic_elpd = float(waic.elpd_waic)
        waic_se = float(waic.se)
        loo_elpd = float(loo.elpd_loo)
        loo_se = float(loo.se)
        waic_p = int(waic.p_waic)
        loo_p = int(loo.p_loo)
    except Exception as e:
        print(f"  WAIC/LOO failed: {e}")
        waic_elpd = waic_se = loo_elpd = loo_se = waic_p = loo_p = None

    # ----- Report -----
    report = {
        "subject_id": subject_id,
        "model": args.model,
        "n_go": int(len(go_df)),
        "n_stop": int(len(stop_df)),
        "r_hat_max": rhat_max,
        "r_hat_ok": rhat_ok,
        "waic_elpd": waic_elpd,
        "waic_se": waic_se,
        "waic_p": waic_p,
        "loo_elpd": loo_elpd,
        "loo_se": loo_se,
        "loo_p": loo_p,
    }
    if rhat is not None:
        report["r_hat_by_param"] = {str(k): float(v) for k, v in rhat.items()}

    report_path = os.path.join(args.output_dir, f"{subject_id}_{args.model}_fit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")

    # Console summary
    print("\n--- Goodness-of-fit summary ---")
    if rhat_max is not None:
        print(f"  R̂ max: {rhat_max:.4f}  {'OK (≤1.01)' if rhat_ok else 'WARNING (>1.01)'}")
    if waic_elpd is not None:
        print(f"  WAIC: elpd_waic = {waic_elpd:.2f} (se = {waic_se:.2f}), p_waic = {waic_p}")
    if loo_elpd is not None:
        print(f"  LOO:  elpd_loo  = {loo_elpd:.2f} (se = {loo_se:.2f}), p_loo = {loo_p}")
    print("  (Lower WAIC/LOO elpd = worse; compare models by more negative = worse fit.)")
    print("  Figures: PPC+QP overlap + R̂ bar plot. Check QP plot for slow/fast tail misfit.")


if __name__ == "__main__":
    main()
