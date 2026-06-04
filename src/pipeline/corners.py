# ─────────────────────────────────────────────
#  src/pipeline/corners.py
#  Corner detection, classification, and per-corner stats.
# ─────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import CORNER_SPEED, SILVERSTONE_CORNERS


# ── 1. Corner detection ───────────────────────────────────────────────────────

def detect_corners(
    tel: pd.DataFrame,
    prominence: float = 15.0,
    width: int = 5,
) -> pd.DataFrame:
    """
    Detect corners by finding local minima in the smoothed speed trace.

    Parameters
    ----------
    tel        : cleaned telemetry with 'Speed_smooth' and 'Distance' columns
    prominence : minimum speed drop (km/h) for a valley to count as a corner
    width      : minimum width (samples) of the speed valley

    Returns
    -------
    DataFrame with one row per detected corner:
        Distance_m, MinSpeed, CornerType, Label (Silverstone name if close)
    """
    speed = tel["Speed_smooth"].values
    dist  = tel["Distance"].values

    # find_peaks on *inverted* speed → finds minima
    valleys, props = find_peaks(-speed, prominence=prominence, width=width)

    if len(valleys) == 0:
        return pd.DataFrame(columns=["idx", "Distance_m", "MinSpeed", "CornerType", "Label"])

    records = []
    for idx in valleys:
        min_speed = speed[idx]
        d         = dist[idx]
        ctype     = _classify_speed(min_speed)
        label     = _nearest_silverstone_name(d)
        records.append({
            "idx":        idx,
            "Distance_m": d,
            "MinSpeed":   min_speed,
            "CornerType": ctype,
            "Label":      label,
        })

    return pd.DataFrame(records)


def _classify_speed(speed_kmh: float) -> str:
    for category, (lo, hi) in CORNER_SPEED.items():
        if lo <= speed_kmh < hi:
            return category
    return "fast"


def _nearest_silverstone_name(distance_m: float, tolerance: float = 200.0) -> str:
    """Return the Silverstone corner name if within ``tolerance`` metres."""
    best_name = ""
    best_dist = float("inf")
    for name, ref_d in SILVERSTONE_CORNERS.items():
        if abs(distance_m - ref_d) < best_dist:
            best_dist = abs(distance_m - ref_d)
            best_name = name
    return best_name if best_dist <= tolerance else ""


# ── 2. Per-corner statistics ──────────────────────────────────────────────────

def corner_stats(
    tel: pd.DataFrame,
    corners: pd.DataFrame,
    window_m: float = 150.0,
) -> pd.DataFrame:
    """
    For each detected corner, extract statistics from the telemetry window
    surrounding the apex (±window_m metres).

    Returns a DataFrame with one row per corner, including:
        MinSpeed, MeanSpeed, BrakingDistance_m, ExitAccel_ms2,
        MaxLateralG, ThrottleApplicationDist_m, CornerType, Label
    """
    rows = []
    dist = tel["Distance"].values

    for _, corner in corners.iterrows():
        apex_d = corner["Distance_m"]
        mask   = (dist >= apex_d - window_m) & (dist <= apex_d + window_m)
        seg    = tel[mask]

        if len(seg) < 5:
            continue

        # Braking distance: from first brake press to speed minimum
        braking_dist = _braking_distance(seg, apex_d)

        # Exit acceleration: mean acceleration in second half of window
        exit_mask = dist[mask] >= apex_d
        exit_seg  = seg[exit_mask] if exit_mask.any() else seg
        exit_accel = exit_seg["Acceleration"].clip(lower=0).mean() if "Acceleration" in seg.columns else np.nan

        # Throttle application distance from apex
        throttle_dist = _throttle_application_distance(seg, apex_d)

        rows.append({
            "Distance_m":              corner["Distance_m"],
            "CornerType":              corner["CornerType"],
            "Label":                   corner["Label"],
            "MinSpeed":                seg["Speed"].min(),
            "MeanSpeed":               seg["Speed"].mean(),
            "BrakingDistance_m":       braking_dist,
            "ExitAccel_ms2":           exit_accel,
            "MaxLateralG":             seg["LateralG"].max() if "LateralG" in seg.columns else np.nan,
            "ThrottleApplicationDist": throttle_dist,
        })

    return pd.DataFrame(rows)


def _braking_distance(seg: pd.DataFrame, apex_dist: float) -> float:
    """Distance from first brake press to apex (metres)."""
    if "BrakePoint" not in seg.columns or "Distance" not in seg.columns:
        return np.nan
    entry = seg[seg["Distance"] < apex_dist]
    brakes = entry[entry["BrakePoint"]]
    if brakes.empty:
        return np.nan
    brake_start = brakes["Distance"].iloc[-1]   # last brake start before apex
    return float(apex_dist - brake_start)


def _throttle_application_distance(seg: pd.DataFrame, apex_dist: float) -> float:
    """Distance past the apex where full throttle is first applied (metres)."""
    if "ThrottlePoint" not in seg.columns:
        return np.nan
    exit_seg = seg[seg["Distance"] >= apex_dist]
    full_t   = exit_seg[exit_seg["ThrottlePoint"]]
    if full_t.empty:
        return np.nan
    return float(full_t["Distance"].iloc[0] - apex_dist)


# ── 3. Comparative corner table ───────────────────────────────────────────────

def compare_corner_stats(
    stats_a: pd.DataFrame,
    stats_b: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    merge_tolerance: float = 100.0,
) -> pd.DataFrame:
    """
    Merge corner stats from two drivers on nearest-distance match and compute
    per-corner deltas.

    Returns a wide DataFrame with _A / _B suffixes and delta columns.
    """
    stats_a = stats_a.copy().rename(
        columns={c: f"{c}_{driver_a}" for c in stats_a.columns if c not in ["Label", "CornerType", "Distance_m"]}
    )
    stats_b = stats_b.copy().rename(
        columns={c: f"{c}_{driver_b}" for c in stats_b.columns if c not in ["Label", "CornerType", "Distance_m"]}
    )

    # Merge on nearest distance
    merged_rows = []
    for _, row_a in stats_a.iterrows():
        dists = (stats_b["Distance_m"] - row_a["Distance_m"]).abs()
        nearest_idx = dists.idxmin()
        if dists[nearest_idx] <= merge_tolerance:
            row_b = stats_b.loc[nearest_idx]
            combined = {**row_a.to_dict(), **row_b.to_dict()}
            merged_rows.append(combined)

    if not merged_rows:
        return pd.DataFrame()

    cmp = pd.DataFrame(merged_rows)

    # Speed delta: positive → driver A faster through corner
    cmp["SpeedDelta"] = cmp[f"MinSpeed_{driver_a}"] - cmp[f"MinSpeed_{driver_b}"]
    cmp["Winner"]     = np.where(cmp["SpeedDelta"] > 0, driver_a, driver_b)

    return cmp.drop_duplicates(subset=["Distance_m"]).reset_index(drop=True)
