# ─────────────────────────────────────────────
#  src/features/aero.py
#  Extract aero-proxy features from a single lap's telemetry.
#  These form the backbone of the XGBoost feature matrix.
# ─────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CORNER_SPEED


# ─────────────────────────────────────────────────────────────────────────────
#  Core feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_aero_features(
    tel: pd.DataFrame,
    corners: pd.DataFrame,
    lap_time_s: float | None = None,
    driver: str | None = None,
    lap_number: int | None = None,
) -> dict:
    """
    Extract a flat dict of aero-proxy features from telemetry + corner data
    for a single lap.

    Parameters
    ----------
    tel        : fully processed telemetry DataFrame (with derived channels)
    corners    : corner DataFrame from ``detect_corners``
    lap_time_s : lap time in seconds (used as label in ML; optional here)
    driver     : driver code string (metadata)
    lap_number : race/quali lap number (metadata)

    Returns
    -------
    dict mapping feature name → scalar value
    """
    feats: dict = {}

    # ── Metadata ──────────────────────────────────────────────────────────────
    if driver is not None:
        feats["driver"] = driver
    if lap_number is not None:
        feats["lap_number"] = lap_number
    if lap_time_s is not None:
        feats["lap_time_s"] = lap_time_s

    # ── 1. Top speed (low drag / low downforce proxy) ─────────────────────────
    # Red Bull's low-downforce setup should dominate this metric on straights
    feats["top_speed"]         = tel["Speed"].max()
    feats["speed_p95"]         = tel["Speed"].quantile(0.95)   # robust version

    # Hangar-straight window: ~2200–2700 m (Silverstone lap ~6km)
    straight = tel[(tel["Distance"] >= 2200) & (tel["Distance"] <= 2700)]
    feats["hangar_straight_top_speed"] = straight["Speed"].max() if len(straight) > 0 else np.nan

    # ── 2. Corner speed metrics (high downforce proxy) ────────────────────────
    # McLaren's high downforce = higher minimum speeds in fast corners
    for ctype in ["slow", "medium", "fast"]:
        subset = corners[corners["CornerType"] == ctype]
        if subset.empty:
            feats[f"min_speed_{ctype}_mean"]  = np.nan
            feats[f"min_speed_{ctype}_min"]   = np.nan
        else:
            feats[f"min_speed_{ctype}_mean"]  = subset["MinSpeed"].mean()
            feats[f"min_speed_{ctype}_min"]   = subset["MinSpeed"].min()

    # ── 3. DRS metrics ────────────────────────────────────────────────────────
    if "DRS_open" in tel.columns:
        drs_open  = tel[tel["DRS_open"]]
        drs_shut  = tel[~tel["DRS_open"]]
        feats["drs_open_mean_speed"]  = drs_open["Speed"].mean()  if len(drs_open)  > 0 else np.nan
        feats["drs_shut_mean_speed"]  = drs_shut["Speed"].mean()  if len(drs_shut)  > 0 else np.nan
        feats["drs_speed_delta"]      = feats["drs_open_mean_speed"] - feats["drs_shut_mean_speed"]
        feats["drs_open_fraction"]    = len(drs_open) / len(tel)

    # ── 4. Braking metrics (aerodynamic braking effectiveness) ───────────────
    if "BrakingDistance_m" in corners.columns:
        feats["braking_dist_mean"]     = corners["BrakingDistance_m"].mean()
        feats["braking_dist_slow"]     = corners[corners["CornerType"] == "slow"]["BrakingDistance_m"].mean()
        feats["braking_dist_fast"]     = corners[corners["CornerType"] == "fast"]["BrakingDistance_m"].mean()

    # ── 5. Traction / exit acceleration (mechanical + aero grip) ─────────────
    if "ExitAccel_ms2" in corners.columns:
        feats["exit_accel_mean"]  = corners["ExitAccel_ms2"].mean()
        feats["exit_accel_fast"]  = corners[corners["CornerType"] == "fast"]["ExitAccel_ms2"].mean()
        feats["exit_accel_slow"]  = corners[corners["CornerType"] == "slow"]["ExitAccel_ms2"].mean()

    # ── 6. Throttle application distance from apex ────────────────────────────
    # Shorter = more confidence = more mechanical / aero grip
    if "ThrottleApplicationDist" in corners.columns:
        feats["throttle_app_dist_mean"] = corners["ThrottleApplicationDist"].mean()
        feats["throttle_app_dist_slow"] = corners[corners["CornerType"] == "slow"]["ThrottleApplicationDist"].mean()

    # ── 7. Lateral G (high downforce → higher lateral G in fast corners) ──────
    if "MaxLateralG" in corners.columns:
        feats["lateral_g_fast_mean"]  = corners[corners["CornerType"] == "fast"]["MaxLateralG"].mean()
        feats["lateral_g_slow_mean"]  = corners[corners["CornerType"] == "slow"]["MaxLateralG"].mean()

    # ── 8. Speed consistency ──────────────────────────────────────────────────
    # Lower std in medium/fast corners → more planted car
    for ctype in ["medium", "fast"]:
        subset = corners[corners["CornerType"] == ctype]
        feats[f"speed_std_{ctype}"] = subset["MinSpeed"].std() if len(subset) > 1 else np.nan

    # ── 9. Engine / power metrics ─────────────────────────────────────────────
    if "RPM" in tel.columns:
        feats["mean_rpm"]     = tel["RPM"].mean()
        feats["max_rpm"]      = tel["RPM"].max()

    # ── 10. Throttle trace summary ────────────────────────────────────────────
    if "Throttle" in tel.columns:
        feats["full_throttle_pct"]  = (tel["Throttle"] >= 95).mean() * 100
        feats["mean_throttle"]      = tel["Throttle"].mean()

    # ── 11. Gear usage ────────────────────────────────────────────────────────
    if "nGear" in tel.columns:
        feats["mean_gear"]  = tel["nGear"].mean()
        feats["max_gear"]   = tel["nGear"].max()

    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  Batch builder: build feature matrix from multiple laps
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(lap_feature_list: list[dict]) -> pd.DataFrame:
    """
    Convert a list of per-lap feature dicts into a tidy DataFrame.
    NaN values are kept; the caller decides how to impute.
    """
    return pd.DataFrame(lap_feature_list)


# ─────────────────────────────────────────────────────────────────────────────
#  Aero signature: compact 6-feature fingerprint per driver
# ─────────────────────────────────────────────────────────────────────────────

RADAR_FEATURES = [
    ("top_speed",              "Top Speed"),
    ("min_speed_fast_mean",    "Fast-Corner Min Speed"),
    ("drs_speed_delta",        "DRS Gain"),
    ("braking_dist_mean",      "Braking Distance"),
    ("lateral_g_fast_mean",    "Fast-Corner Lateral G"),
    ("exit_accel_mean",        "Exit Acceleration"),
]


def aero_signature(feature_row: pd.Series) -> dict[str, float]:
    """
    Return the 6-feature aero-signature dict for a single row of the
    feature matrix. Used for the radar chart.
    """
    return {
        label: float(feature_row.get(col, np.nan))
        for col, label in RADAR_FEATURES
    }
