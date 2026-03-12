import pandas as pd


REQUIRED_COLUMNS = [
    "subject_id",
    "trial_type",  # "go" or "stop"
    "rt",  # reaction time in seconds; NaN or 0 for no response
    "response",  # "correct", "incorrect", "inhibit", "respond"
    "ssd",  # stop-signal delay in seconds; 0 or NaN for go
]


def load_subject_csv(path, subject_id=None):
    """
    Load a single-subject CSV and ensure required columns exist.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    subject_id : optional
        If given, will filter or set the subject_id column.
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

