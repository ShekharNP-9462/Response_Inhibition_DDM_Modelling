import numpy as np


def integration_ssrt(go_rts, stop_ssd, stop_success_flags, stop_fail_flags):
    """
    Compute SSRT using the integration method (Logan & Cowan, 1984).

    Parameters
    ----------
    go_rts : array_like
        Reaction times (seconds) from correct go trials used for the
        integration distribution (e.g. MS_GO go trials).
    stop_ssd : array_like
        Stop-signal delays (seconds) for all stop trials.
    stop_success_flags : array_like of bool
        True for successful inhibitions.
    stop_fail_flags : array_like of bool
        True for failed inhibitions (signal-respond).

    Returns
    -------
    ssrt : float
        SSRT estimate in seconds.
    details : dict
        Dictionary with intermediate quantities:
        - n_go, n_stop, n_fail
        - p_respond
        - rt_integral
        - mean_ssd
    """
    go_rts = np.asarray(go_rts, dtype=float)
    stop_ssd = np.asarray(stop_ssd, dtype=float)
    stop_success_flags = np.asarray(stop_success_flags, dtype=bool)
    stop_fail_flags = np.asarray(stop_fail_flags, dtype=bool)

    # Filter valid data
    go_rts = go_rts[np.isfinite(go_rts)]
    stop_mask = np.isfinite(stop_ssd)
    stop_ssd = stop_ssd[stop_mask]
    stop_success_flags = stop_success_flags[stop_mask]
    stop_fail_flags = stop_fail_flags[stop_mask]

    n_go = go_rts.size
    n_stop = stop_ssd.size
    n_fail = int(stop_fail_flags.sum())

    if n_go == 0 or n_stop == 0:
        raise ValueError("Need at least one valid go RT and stop trial for SSRT.")

    p_respond = n_fail / n_stop

    # Integration index
    go_sorted = np.sort(go_rts)
    idx = int(np.ceil(p_respond * n_go)) - 1
    idx = max(0, min(idx, n_go - 1))
    rt_integral = go_sorted[idx]

    mean_ssd = float(np.mean(stop_ssd))
    ssrt = rt_integral - mean_ssd

    details = {
        "n_go": n_go,
        "n_stop": n_stop,
        "n_fail": n_fail,
        "p_respond": p_respond,
        "rt_integral": rt_integral,
        "mean_ssd": mean_ssd,
    }
    return ssrt, details

