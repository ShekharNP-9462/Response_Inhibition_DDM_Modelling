"""
Single-subject full diffusion model (DDM) for Rashi pre-cued data.

Go RTs: shifted Wald (drift v, boundary a, non-decision time Ter).
Stop: ex-Gaussian SSRT in race with go DDM.
Outputs: v (drift rate), a (boundary), ter (non-decision time), plus SSRT ex-Gaussian params.
Includes elapsed-time and ETA logging.
"""
import os as _os
if "PYTENSOR_FLAGS" not in _os.environ:
    _os.environ["PYTENSOR_FLAGS"] = "cxx="

import argparse
import os
import time

import arviz as az
import pymc as pm

from ddm_beests.io import prepare_rashi_single_subject_for_model
from ddm_beests.ddm_model import build_single_subject_ddm_model


def _make_progress_callback(total_steps, start_time, log_interval=100):
    """Return a callback that logs elapsed time and ETA every log_interval steps."""
    state = {"steps_done": 0}

    def callback(*args, **kwargs):
        state["steps_done"] += 1
        n = state["steps_done"]
        if n % log_interval == 0 or n == total_steps:
            elapsed = time.perf_counter() - start_time
            if n > 0:
                rate = n / elapsed
                remaining_sec = (total_steps - n) / rate
                eta_str = f"{remaining_sec / 60:.1f} min remaining"
            else:
                eta_str = "estimating..."
            print(f"  [DDM] steps {n}/{total_steps} | elapsed {elapsed / 60:.1f} min | {eta_str}")
    return callback


def main():
    parser = argparse.ArgumentParser(
        description="Fit single-subject full DDM (drift, boundary, Ter + SSRT) to Rashi data.",
    )
    parser.add_argument(
        "subject_dir",
        type=str,
        help="Path to subject directory (e.g. .../Rashi_Dataset/EF_Y_01).",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--n-mc", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=100, help="Steps between ETA logs.")
    args = parser.parse_args()

    subject_dir = os.path.abspath(args.subject_dir)
    total_steps = (args.tune + args.draws) * args.chains

    t_start = time.perf_counter()
    print(f"[DDM] Started at {time.strftime('%Y-%m-%d %H:%M:%S')} (total steps: {total_steps})")

    go_df, stop_df = prepare_rashi_single_subject_for_model(subject_dir)
    if go_df.empty or stop_df.empty:
        raise ValueError("Go or stop data is empty for this subject.")

    model = build_single_subject_ddm_model(go_df, stop_df, n_mc=args.n_mc)
    callback = _make_progress_callback(total_steps, t_start, args.log_interval)

    with model:
        step = pm.Slice()
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            step=step,
            callback=callback,
            progressbar=True,
        )

    elapsed = time.perf_counter() - t_start
    print(f"[DDM] Finished at {time.strftime('%Y-%m-%d %H:%M:%S')} | total elapsed: {elapsed / 60:.1f} min")
    print(az.summary(
        trace,
        var_names=["v", "a", "ter", "mu_ssrt", "sigma_ssrt", "tau_ssrt"],
    ))

    if args.output:
        az.to_netcdf(trace, args.output)
        print(f"[DDM] Saved trace to {args.output}")


if __name__ == "__main__":
    main()
