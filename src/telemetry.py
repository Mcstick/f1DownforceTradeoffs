# ─────────────────────────────────────────────
#  src/pipeline/telemetry.py
#  Telemetry cleaning, alignment, and derived channel computation.
# ─────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import MIN_VALID_LAP_TIME_S, MAX_VALID_LAP_TIME_S


# ── 1. Cleaning ───────────────────────────────────────────────────────────────

def clean_telemetry(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sanity-clean on a raw telemetry DataFrame.

    - Drop rows where Speed is NaN or <= 0
    - Clip Throttle to [0, 100]
    - Ensure Distance is monotonically increasing (drop backwards jumps)
    - Smooth Speed with a Savitzky-Golay filter (reduces GPS noise)
    """
    tel = tel.copy()

    # Speed sanity
    tel = tel[tel["Speed"].notna() & (tel["Speed"] > 0)]

    # Throttle clip
    if "Throttle" in tel.columns:
        tel["Throttle"] = tel["Throttle"].clip(0, 100)

    # Monotonic distance – keep only forward-progressing rows
    if "Distance" in tel.columns:
        tel = tel[tel["Distance"].diff().fillna(1) >= 0]

    # Smooth speed
    if len(tel) > 11:
        tel["Speed_smooth"] = savgol_filter(tel["Speed"], window_length=11, polyorder=3)
    else:
        tel["Speed_smooth"] = tel["Speed"]

    return tel.reset_index(drop=True)


# ── 2. Derived channels ───────────────────────────────────────────────────────

def add_derived_channels(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional physics-derived channels:

    - ``Acceleration``  : longitudinal acceleration (m/s²) via Δspeed/Δtime
    - ``Deceleration``  : negative acceleration (braking zones), positive values
    - ``LateralG_proxy``: estimated lateral g-force  = v² / (r * 9.81)
                          where r is estimated from the rate of heading change
    - ``BrakePoint``    : boolean, True at the start of each braking zone
    - ``ThrottlePoint`` : boolean, True at the start of each full-throttle zone
    """
    tel = tel.copy()

    # Time in seconds from lap start
    if "Time" in tel.columns and pd.api.types.is_timedelta64_dtype(tel["Time"]):
        tel["Time_s"] = tel["Time"].dt.total_seconds()
    elif "Time_s" not in tel.columns:
        # Fallback: assume 10 Hz sampling
        tel["Time_s"] = np.arange(len(tel)) * 0.1

    dt = tel["Time_s"].diff().replace(0, np.nan)
    v  = tel["Speed"] / 3.6   # km/h → m/s

    tel["Acceleration"]  = (v.diff() / dt).fillna(0)
    tel["Deceleration"]  = (-tel["Acceleration"]).clip(lower=0)

    # Lateral G proxy from track X/Y heading change
    if "X" in tel.columns and "Y" in tel.columns:
        dx     = tel["X"].diff()
        dy     = tel["Y"].diff()
        ds     = np.sqrt(dx**2 + dy**2).replace(0, np.nan)

        heading      = np.arctan2(dy, dx)
        dheading_dt  = heading.diff() / dt           # rad/s  (yaw rate)
        # lateral acc = v * omega
        tel["LateralAcc"] = (v * dheading_dt.abs()).fillna(0)
        tel["LateralG"]   = tel["LateralAcc"] / 9.81
    else:
        tel["LateralG"] = np.nan

    # Brake / throttle event markers
    if "Brake" in tel.columns:
        brake_bool          = tel["Brake"] > 50 if tel["Brake"].max() > 1 else tel["Brake"].astype(bool)
        tel["BrakePoint"]   = brake_bool & (~brake_bool.shift(1).fillna(False))

    if "Throttle" in tel.columns:
        full_throttle           = tel["Throttle"] >= 95
        tel["ThrottlePoint"]    = full_throttle & (~full_throttle.shift(1).fillna(False))

    return tel


# ── 3. Distance alignment ─────────────────────────────────────────────────────

def align_by_distance(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    resolution: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Interpolate two telemetry DataFrames onto a common distance axis so they
    can be directly compared point-by-point.

    Parameters
    ----------
    tel_a, tel_b : cleaned telemetry DataFrames (must have 'Distance' column)
    resolution   : distance step in metres (default 5 m)

    Returns
    -------
    Two DataFrames on the same distance grid.
    """
    d_min = max(tel_a["Distance"].min(), tel_b["Distance"].min())
    d_max = min(tel_a["Distance"].max(), tel_b["Distance"].max())

    common_dist = np.arange(d_min, d_max, resolution)

    numeric_cols_a = tel_a.select_dtypes(include=np.number).columns.tolist()
    numeric_cols_b = tel_b.select_dtypes(include=np.number).columns.tolist()

    aligned_a = _interp_to_grid(tel_a, common_dist, numeric_cols_a)
    aligned_b = _interp_to_grid(tel_b, common_dist, numeric_cols_b)

    return aligned_a, aligned_b


def _interp_to_grid(
    tel: pd.DataFrame,
    grid: np.ndarray,
    cols: list[str],
) -> pd.DataFrame:
    """Linear interpolation of all numeric columns onto ``grid``."""
    out = {"Distance": grid}
    for col in cols:
        if col == "Distance":
            continue
        out[col] = np.interp(grid, tel["Distance"].values, tel[col].values)

    # Carry through non-numeric metadata (Driver, Session)
    for meta in ["Driver", "Session"]:
        if meta in tel.columns:
            out[meta] = tel[meta].iloc[0]

    return pd.DataFrame(out)


# ── 4. Delta time calculation ─────────────────────────────────────────────────

def compute_delta_time(
    tel_ref: pd.DataFrame,
    tel_comp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cumulative lap-delta (seconds) between two aligned telemetry frames.

    Positive delta  → comp driver is ahead of ref at that point
    Negative delta  → comp driver is behind ref

    Both DataFrames must already be aligned on the same distance grid
    (output of ``align_by_distance``).

    Returns the ref DataFrame with an additional ``Delta_s`` column.
    """
    assert len(tel_ref) == len(tel_comp), \
        "Telemetry frames must be aligned before computing delta."

    v_ref  = tel_ref["Speed"].values  / 3.6   # m/s
    v_comp = tel_comp["Speed"].values / 3.6

    dist   = tel_ref["Distance"].values
    ds     = np.diff(dist, prepend=dist[0])

    # Time to cover each ds segment for each driver
    dt_ref  = np.where(v_ref  > 0, ds / v_ref,  0)
    dt_comp = np.where(v_comp > 0, ds / v_comp, 0)

    cumulative_delta = np.cumsum(dt_ref - dt_comp)

    result              = tel_ref.copy()
    result["Delta_s"]   = cumulative_delta
    return result


# ── 5. Lap filtering ──────────────────────────────────────────────────────────

def filter_valid_race_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outlier laps from a race lap DataFrame.
    Keeps laps within [MIN_VALID_LAP_TIME_S, MAX_VALID_LAP_TIME_S].
    Also drops laps tagged with a TrackStatus that implies SC/VSC.
    """
    mask = (
        laps["LapTime_s"].between(MIN_VALID_LAP_TIME_S, MAX_VALID_LAP_TIME_S)
    )

    # Drop safety-car affected laps if TrackStatus column exists
    if "TrackStatus" in laps.columns:
        normal_status = laps["TrackStatus"].isin(["1", "2", 1, 2])
        mask = mask & normal_status

    return laps[mask].reset_index(drop=True)
