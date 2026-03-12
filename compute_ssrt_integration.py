import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ddm_beests.io import (
    load_rashi_subject_dir,
    rashi_add_derived_columns,
    rashi_to_generic_stop_signal,
)
from ddm_beests.integration_ssrt import integration_ssrt


def infer_group_from_subject_id(subject_id):
    """
    Infer age group (Y or O) from a subject_id like 'EF_Y_01'.
    """
    m = re.search(r"EF_([YO])_", subject_id)
    if m:
        tag = m.group(1)
        return "young" if tag == "Y" else "old"
    return "unknown"


def compute_subject_ssrt(subject_dir):
    raw = load_rashi_subject_dir(subject_dir)
    raw = rashi_add_derived_columns(raw)
    _ = rashi_to_generic_stop_signal(raw)  # kept for future use; not strictly needed here

    subject_id = raw["subject_id"].iloc[0]
    group = infer_group_from_subject_id(subject_id)

    # For a pre-cued task, SSRT is typically based on MS_GO trials,
    # since stop signals occur in that context.
    ms_go_mask = raw["is_valid_go_rt"] & raw["is_ms_go"]
    go_rts_ms = raw.loc[ms_go_mask, "ReactionTime"].to_numpy()

    stop_mask = raw["is_stop"]
    stop_ssd = raw.loc[stop_mask, "Delay"].to_numpy()
    stop_success = raw.loc[stop_mask, "is_stop_success"].to_numpy()
    stop_fail = raw.loc[stop_mask, "is_stop_fail"].to_numpy()

    ssrt_ms, details_ms = integration_ssrt(go_rts_ms, stop_ssd, stop_success, stop_fail)

    # Optionally, also compute SSRT using all valid go RTs as a rough check
    all_go_rts = raw.loc[raw["is_valid_go_rt"], "ReactionTime"].to_numpy()
    ssrt_all, details_all = integration_ssrt(all_go_rts, stop_ssd, stop_success, stop_fail)

    # Data-cleaning metrics (per Verbruggen et al., Congdon et al., Matzke et al.)
    # 1. Failed stop vs. go RT: Mean(FS_RT) should NOT exceed Mean(Go_RT)
    fs_rts = raw.loc[raw["is_stop_fail"], "ReactionTime"].to_numpy()
    mean_fs_rt = float(np.nanmean(fs_rts)) if fs_rts.size > 0 else np.nan
    mean_go_rt = float(np.nanmean(all_go_rts)) if all_go_rts.size > 0 else np.nan
    failed_stop_vs_go_ok = not (np.isfinite(mean_fs_rt) and np.isfinite(mean_go_rt) and (mean_fs_rt > mean_go_rt))

    # 2. Inhibition rate: P(Inhibit) should be between 25% and 75%
    n_stop = int(details_ms["n_stop"])
    n_success = int(stop_success.sum())
    inhibition_rate = n_success / n_stop if n_stop > 0 else np.nan
    inhibition_rate_ok = np.isfinite(inhibition_rate) and (0.25 <= inhibition_rate <= 0.75)

    # 3. Go accuracy: >= 60%
    n_go_total = int(raw["is_go"].sum())
    n_go_correct = int((raw["is_go"] & raw["is_correct"]).sum())
    go_accuracy = n_go_correct / n_go_total if n_go_total > 0 else np.nan
    go_accuracy_ok = np.isfinite(go_accuracy) and (go_accuracy >= 0.60)

    # 4. Go omissions: <= 20% of go trials (user-specified)
    n_go_omissions = int(raw["is_go_omission"].sum())
    go_omission_rate = n_go_omissions / n_go_total if n_go_total > 0 else np.nan
    go_omissions_ok = np.isfinite(go_omission_rate) and (go_omission_rate <= 0.20)

    passed_cleaning = (
        failed_stop_vs_go_ok
        and inhibition_rate_ok
        and go_accuracy_ok
        and go_omissions_ok
    )

    # Additional counts for bookkeeping
    n_all_trials = int(len(raw))
    n_anticipations = int(raw["is_anticipation"].sum())
    # Go and stop failures (errors) separately
    n_fail_stop = int(stop_fail.sum())
    n_fail_go = int((raw["is_go"] & (~raw["is_correct"])).sum())

    return {
        "subject_id": subject_id,
        "group": group,
        "ssrt_ms": ssrt_ms,
        "ssrt_ms_p_respond": details_ms["p_respond"],
        "ssrt_ms_mean_ssd": details_ms["mean_ssd"],
        "n_go_ms": details_ms["n_go"],
        "n_stop": details_ms["n_stop"],
        "n_fail": details_ms["n_fail"],
        "ssrt_all": ssrt_all,
        "ssrt_all_p_respond": details_all["p_respond"],
        "ssrt_all_mean_ssd": details_all["mean_ssd"],
        "n_go_all": details_all["n_go"],
        "mean_fs_rt": mean_fs_rt,
        "mean_go_rt": mean_go_rt,
        "failed_stop_vs_go_ok": failed_stop_vs_go_ok,
        "inhibition_rate": inhibition_rate,
        "inhibition_rate_ok": inhibition_rate_ok,
        "go_accuracy": go_accuracy,
        "go_accuracy_ok": go_accuracy_ok,
        "go_omission_rate": go_omission_rate,
        "go_omissions_ok": go_omissions_ok,
        "passed_cleaning": passed_cleaning,
        "n_all_trials": n_all_trials,
        "n_anticipations": n_anticipations,
        "n_fail_stop": n_fail_stop,
        "n_fail_go": n_fail_go,
    }, go_rts_ms, all_go_rts


def main():
    parser = argparse.ArgumentParser(
        description="Compute integration-method SSRTs for all subjects in the Rashi dataset."
    )
    parser.add_argument(
        "dataset_root",
        type=str,
        help="Path to 'Rashi_Dataset' directory containing EF_* subject folders.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="EF_",
        help="Prefix pattern for subject folders (default: EF_).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save a CSV summary of SSRT estimates.",
    )

    args = parser.parse_args()

    root = os.path.abspath(args.dataset_root)
    subject_dirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if d.startswith(args.pattern) and os.path.isdir(os.path.join(root, d))
    ]

    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders starting with '{args.pattern}' found in {root}")

    rows = []
    young_go_rts = []
    old_go_rts = []

    for sd in sorted(subject_dirs):
        try:
            res, go_rts_ms, all_go_rts = compute_subject_ssrt(sd)
            rows.append(res)
            print(
                f"{res['subject_id']:10s} ({res['group']:5s})  "
                f"SSRT_MS={res['ssrt_ms']*1000:.1f} ms  "
                f"SSRT_all={res['ssrt_all']*1000:.1f} ms  "
                f"p(respond|signal)={res['ssrt_ms_p_respond']:.3f}  "
                f"n_go_MS={res['n_go_ms']}, n_stop={res['n_stop']}  "
                f"passed_cleaning={res['passed_cleaning']}"
            )

            # Collect go RTs for plotting (only for subjects passing cleaning)
            if res["passed_cleaning"]:
                if res["group"] == "young":
                    young_go_rts.extend(go_rts_ms.tolist())
                elif res["group"] == "old":
                    old_go_rts.extend(go_rts_ms.tolist())
        except Exception as e:
            print(f"Error processing {sd}: {e}")

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Save CSV if requested
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved SSRT summary to {args.output_csv}")

    # Create preliminary plots comparing SSRT and go RT distributions between young and old
    included = df[df["passed_cleaning"] & df["group"].isin(["young", "old"])]
    if included.empty:
        print("No subjects passed cleaning; skipping plotting.")
        return

    young_ssrt = (included.loc[included["group"] == "young", "ssrt_ms"] * 1000.0).to_numpy()
    old_ssrt = (included.loc[included["group"] == "old", "ssrt_ms"] * 1000.0).to_numpy()

    young_go_rts_arr = np.asarray(young_go_rts) * 1000.0
    old_go_rts_arr = np.asarray(old_go_rts) * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: SSRT distributions by group
    data = [young_ssrt, old_ssrt]
    labels = ["Young", "Old"]
    axes[0].boxplot(data, labels=labels)
    axes[0].set_ylabel("SSRT (ms)")
    axes[0].set_title("SSRT (integration method)")

    # Right: Go RT distributions (MS_GO) by group
    bins = 20
    if young_go_rts_arr.size > 0:
        axes[1].hist(young_go_rts_arr, bins=bins, alpha=0.5, label="Young")
    if old_go_rts_arr.size > 0:
        axes[1].hist(old_go_rts_arr, bins=bins, alpha=0.5, label="Old")
    axes[1].set_xlabel("Go RT (MS_GO, ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Go RT distributions (MS_GO)")
    axes[1].legend()

    fig.suptitle("Young vs Old: SSRT and Go RT (integration method)")
    fig.tight_layout()

    out_path = "ssrt_distribution_old_vs_young.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()

