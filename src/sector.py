# ─────────────────────────────────────────────
#  src/features/sector.py
#  Sector and mini-sector feature engineering.
# ─────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd


# Silverstone sector split distances (metres) – approximate
SECTOR_SPLITS = {
    "S1": (0,    2000),
    "S2": (2000, 4200),
    "S3": (4200, 6626),   # ~full lap distance
}

# Mini-sector width in metres
MINI_SECTOR_WIDTH = 250


# ── 1. Sector-level features ──────────────────────────────────────────────────

def sector_features(tel: pd.DataFrame) -> dict:
    """
    Compute mean speed, max speed, full-throttle %, and top gear for each
    of the three Silverstone sectors.
    """
    feats = {}
    for name, (d_start, d_end) in SECTOR_SPLITS.items():
        seg = tel[(tel["Distance"] >= d_start) & (tel["Distance"] < d_end)]
        if seg.empty:
            feats.update({
                f"{name}_mean_speed": np.nan,
                f"{name}_max_speed":  np.nan,
                f"{name}_full_thr":   np.nan,
            })
            continue

        feats[f"{name}_mean_speed"] = seg["Speed"].mean()
        feats[f"{name}_max_speed"]  = seg["Speed"].max()
        feats[f"{name}_full_thr"]   = (seg["Throttle"] >= 95).mean() * 100 \
                                      if "Throttle" in seg.columns else np.nan

    return feats


# ── 2. Mini-sector time simulation ───────────────────────────────────────────

def compute_mini_sector_times(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Divide the lap into ``MINI_SECTOR_WIDTH``-metre bins and compute
    the time each driver spent in each bin (using speed integration).

    Returns a DataFrame:
        MiniSector (int), DistStart_m, DistEnd_m, Time_s
    """
    max_dist = tel["Distance"].max()
    bins     = np.arange(0, max_dist + MINI_SECTOR_WIDTH, MINI_SECTOR_WIDTH)

    rows = []
    for i in range(len(bins) - 1):
        d_lo, d_hi = bins[i], bins[i + 1]
        seg = tel[(tel["Distance"] >= d_lo) & (tel["Distance"] < d_hi)]

        if len(seg) < 2:
            rows.append({"MiniSector": i, "DistStart_m": d_lo, "DistEnd_m": d_hi, "Time_s": np.nan})
            continue

        # Integrate time via distance / speed
        v    = seg["Speed"].values / 3.6   # m/s
        d    = seg["Distance"].values
        ds   = np.diff(d)
        vmid = (v[:-1] + v[1:]) / 2
        dt   = np.where(vmid > 0, ds / vmid, 0)
        rows.append({
            "MiniSector": i,
            "DistStart_m": d_lo,
            "DistEnd_m":   d_hi,
            "Time_s":      dt.sum(),
        })

    return pd.DataFrame(rows)


def compare_mini_sectors(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Build a side-by-side mini-sector time comparison.

    Returns DataFrame with columns:
        MiniSector, DistStart_m, Time_A, Time_B, Delta_s, Faster
    """
    ms_a = compute_mini_sector_times(tel_a).rename(columns={"Time_s": f"Time_{driver_a}"})
    ms_b = compute_mini_sector_times(tel_b).rename(columns={"Time_s": f"Time_{driver_b}"})

    merged = ms_a.merge(ms_b, on=["MiniSector", "DistStart_m", "DistEnd_m"])
    merged["Delta_s"] = merged[f"Time_{driver_a}"] - merged[f"Time_{driver_b}"]
    merged["Faster"]  = np.where(merged["Delta_s"] < 0, driver_a, driver_b)

    return merged
