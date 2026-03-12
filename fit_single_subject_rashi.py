import argparse
import os

import arviz as az
import pymc as pm

from ddm_beests.io import prepare_rashi_single_subject_for_model
from ddm_beests.model import build_single_subject_model


def main():
    parser = argparse.ArgumentParser(
        description="Fit a single-subject BEESTS-style model to a Rashi pre-cued stop-signal subject."
    )
    parser.add_argument(
        "subject_dir",
        type=str,
        help="Path to subject directory (e.g. .../Rashi_Dataset/EF_Y_01).",
    )
    parser.add_argument("--draws", type=int, default=1000, help="Number of posterior draws.")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning steps.")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains.")
    parser.add_argument(
        "--n-mc",
        type=int,
        default=500,
        help="Monte Carlo samples per likelihood evaluation for stop trials.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save ArviZ InferenceData (NetCDF).",
    )

    args = parser.parse_args()

    subject_dir = os.path.abspath(args.subject_dir)

    go_df, stop_df = prepare_rashi_single_subject_for_model(subject_dir)

    if go_df.empty or stop_df.empty:
        raise ValueError("Go or stop data is empty for this subject; check preprocessing/cleaning.")

    model = build_single_subject_model(go_df, stop_df, n_mc=args.n_mc)

    with model:
        step = pm.Slice()
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            step=step,
        )

    print(az.summary(trace, var_names=["mu_go", "sigma_go", "tau_go", "mu_ssrt", "sigma_ssrt", "tau_ssrt"]))

    if args.output is not None:
        az.to_netcdf(trace, args.output)


if __name__ == "__main__":
    main()

