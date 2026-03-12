import os
import glob

import pandas as pd


# Generic columns for the abstract stop-signal representation used by the model code
REQUIRED_COLUMNS = [
    "subject_id",
    "trial_type",  # "go" or "stop"
    "rt",  # reaction time in seconds; NaN or 0 for no response
    "response",  # e.g. "correct", "incorrect", "inhibit", "respond"
    "ssd",  # stop-signal delay in seconds; 0 or NaN for go
]


def load_subject_csv(path, subject_id=None):
    """
    Load a generic single-subject CSV and ensure required columns exist.

    This is a simple loader for already preprocessed stop-signal data.
    For the Rashi dataset files, use `load_rashi_subject_dir` instead.
    """
    df = pd.read_csv(path)

    # If subject_id is not present, set it from argument
    if "subject_id" not in df.columns:
        if subject_id is None:
            raise ValueError("CSV has no 'subject_id' column and no subject_id was provided.")
        df["subject_id"] = subject_id

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")

    # Optional: basic cleaning
    df = df.copy()
    # Convert RT and SSD to seconds if they look like ms (heuristic)
    for col in ["rt", "ssd"]:
        if df[col].dropna().mean() > 10:  # probably in ms
            df[col] = df[col] / 1000.0

    return df


def split_go_stop(df, rt_min=0.15, rt_max=3.0):
    """
    Convenience function to split go and stop trials and apply simple RT trimming.
    """
    df = df.copy()
    # Trim implausible RTs on go and failed-stop trials
    mask_rt = df["rt"].between(rt_min, rt_max) | df["rt"].isna()
    df = df.loc[mask_rt]

    go = df[df["trial_type"] == "go"].copy()
    stop = df[df["trial_type"] == "stop"].copy()

    return go, stop


def load_rashi_subject_dir(subject_dir):
    """
    Load and concatenate all 'trialData*.csv' files for a Rashi dataset subject.

    The raw columns are expected to include:
      - 'TrialType', 'Response', 'Correct', 'ReactionTime', 'Delay', ...

    Returns a DataFrame in the *raw* schema, plus a 'subject_id' column
    inferred from the folder name (e.g. EF_Y_01).
    """
    subject_id = os.path.basename(os.path.normpath(subject_dir))
    pattern = os.path.join(subject_dir, "trialData*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No 'trialData*.csv' files found in {subject_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, na_values=["Missing value"])
        df["subject_id"] = subject_id
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    return full


def rashi_add_derived_columns(df):
    """
    Add derived columns (go/stop coding, cleaned RT/SSD) to a raw Rashi DataFrame.
    """
    df = df.copy()

    # Clean RT and Delay; they may already be in seconds, but we keep as-is.
    df["ReactionTime"] = pd.to_numeric(df["ReactionTime"], errors="coerce")
    df["Delay"] = pd.to_numeric(df["Delay"], errors="coerce")

    # Trial type categories
    trial_type = df["TrialType"].astype(str)
    df["is_stop"] = trial_type.str.contains("Stop", case=False, na=False)
    df["is_ms_go"] = trial_type.str.contains("MS_GO", case=False, na=False) & ~df["is_stop"]
    df["is_certain_go"] = trial_type.str.contains("CertainGo", case=False, na=False) & ~df["is_stop"]
    df["is_go"] = (df["is_ms_go"] | df["is_certain_go"]) & ~df["is_stop"]

    # Anticipation / unusable trials from Response column
    resp = df["Response"].astype(str)
    df["is_anticipation"] = resp.str.contains("Anticipation", case=False, na=False)

    # Correct coding: 1 = correct, 0 = incorrect / anticipation (per your description)
    # We keep 'Correct' as-is but also define interpretable flags:
    df["is_correct"] = df["Correct"] == 1

    # Successful inhibition on stop trials: no response, correct == 1
    df["is_stop_success"] = df["is_stop"] & df["is_correct"] & resp.str.contains("NoResponse", case=False, na=False)
    # Failed inhibition: stop trial, incorrect (Correct == 0)
    df["is_stop_fail"] = df["is_stop"] & (~df["is_correct"])

    # Go omissions: go trial with missing RT or explicit NoResponse
    df["is_go_omission"] = df["is_go"] & (
        df["ReactionTime"].isna()
        | resp.str.contains("NoResponse", case=False, na=False)
    )

    # Valid go trials for RT distribution: go, correct, not anticipation, with RT
    df["is_valid_go_rt"] = (
        df["is_go"]
        & df["is_correct"]
        & (~df["is_anticipation"])
        & df["ReactionTime"].notna()
    )

    return df


def rashi_to_generic_stop_signal(df):
    """
    Convert a Rashi raw DataFrame with derived columns into the generic
    stop-signal representation required by the modelling code.
    """
    if "subject_id" not in df.columns:
        raise ValueError("Rashi DataFrame must contain 'subject_id'.")

    # Create a generic trial_type, rt, response, ssd view
    df = df.copy()

    generic = pd.DataFrame(
        {
            "subject_id": df["subject_id"],
            "trial_type": pd.Series(index=df.index, dtype="object"),
            "rt": pd.Series(index=df.index, dtype="float"),
            "response": pd.Series(index=df.index, dtype="object"),
            "ssd": pd.Series(index=df.index, dtype="float"),
        }
    )

    # Go trials
    generic.loc[df["is_go"], "trial_type"] = "go"
    generic.loc[df["is_go"], "rt"] = df.loc[df["is_go"], "ReactionTime"]
    generic.loc[df["is_go"], "response"] = df.loc[df["is_go"], "Response"]
    generic.loc[df["is_go"], "ssd"] = 0.0

    # Stop trials
    generic.loc[df["is_stop"], "trial_type"] = "stop"
    generic.loc[df["is_stop"], "ssd"] = df.loc[df["is_stop"], "Delay"]

    # Successful inhibition: no response, no RT
    generic.loc[df["is_stop_success"], "response"] = "inhibit"
    generic.loc[df["is_stop_success"], "rt"] = pd.NA

    # Failed inhibition: use observed RT and mark as respond
    generic.loc[df["is_stop_fail"], "response"] = "respond"
    generic.loc[df["is_stop_fail"], "rt"] = df.loc[df["is_stop_fail"], "ReactionTime"]

    # Basic RT trimming for generic representation
    # (More specific trimming can still be applied by split_go_stop if desired)
    return generic

