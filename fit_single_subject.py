import argparse

import arviz as az
import pymc as pm

from ddm_beests.io import load_subject_csv, split_go_stop
from ddm_beests.model import build_single_subject_model


def main():
    parser = argparse.ArgumentParser(description="Fit single-subject BEESTS-style stop-signal model with PyMC.")
    parser.add_argument("csv_path", type=str, help="Path to subject CSV file.")
    parser.add_argument("--subject-id", type=str, default=None, help="Subject ID (if not in CSV).")
    parser.add_argument("--draws", type=int, default=1000, help="Number of posterior draws.")
    parser.add_argument("--tune", type=int, default=1000, help="Number of tuning steps.")
    parser.add_argument("--chains", type=int, default=2, help="Number of MCMC chains.")
    parser.add_argument("--target-accept", type=float, default=0.8, help="Target acceptance for samplers that use it.")
    parser.add_argument("--n-mc", type=int, default=500, help="Monte Carlo samples per likelihood evaluation for stop trials.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save ArviZ InferenceData (NetCDF). If omitted, nothing is saved.",
    )

    args = parser.parse_args()

    df = load_subject_csv(args.csv_path, subject_id=args.subject_id)
    go_df, stop_df = split_go_stop(df)

    model = build_single_subject_model(go_df, stop_df, n_mc=args.n_mc)

    with model:
        # Use a generic sampler (Slice or Metropolis) since the likelihood is a black-box DensityDist
        step = pm.Slice()
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            step=step,
            target_accept=args.target_accept,
        )

    print(az.summary(trace, var_names=["mu_go", "sigma_go", "tau_go", "mu_ssrt", "sigma_ssrt", "tau_ssrt"]))

    if args.output is not None:
        az.to_netcdf(trace, args.output)


if __name__ == "__main__":
    main()

