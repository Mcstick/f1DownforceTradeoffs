# ─────────────────────────────────────────────
#  src/viz/plots.py
#  All project visualizations — dark F1-style theme.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from config import DRIVERS, OUTPUT_DIR, FIG_DPI, CORNER_SPEED

log = logging.getLogger(__name__)

# ── Palette helpers ───────────────────────────────────────────────────────────
DRIVER_COLORS = {drv: info["color"] for drv, info in DRIVERS.items()}
CORNER_COLORS = {"slow": "#e74c3c", "medium": "#f39c12", "fast": "#2ecc71"}


def _driver_color(drv: str) -> str:
    return DRIVER_COLORS.get(drv, "#ffffff")


def _save(fig: plt.Figure, name: str) -> Path:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR) / f"{name}.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info("Saved → %s", out)
    return out


def _fmt_laptime(s: float) -> str:
    """Format seconds as M:SS.mmm"""
    m   = int(s // 60)
    rem = s - m * 60
    return f"{m}:{rem:06.3f}"


# ─────────────────────────────────────────────────────────────────────────────
#  1. Speed trace overlay (coloured by corner type)
# ─────────────────────────────────────────────────────────────────────────────

def plot_speed_trace(
    tel_a: pd.DataFrame,
    tel_b: pd.DataFrame,
    corners_a: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    session_label: str = "Fastest Lap",
    save: bool = True,
) -> plt.Figure:
    """
    Dual-driver speed trace overlaid on the same axes, with corner-type
    shading in the background.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [4, 1, 1]},
                             facecolor="#0d0d0d")
    ax_speed, ax_throttle, ax_brake = axes
    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    # ── Corner shading ────────────────────────────────────────────────────────
    for _, corner in corners_a.iterrows():
        col  = CORNER_COLORS.get(corner["CornerType"], "#888")
        apex = corner["Distance_m"]
        for ax in axes:
            ax.axvspan(apex - 80, apex + 80, color=col, alpha=0.08, lw=0)
        # Label first occurrence
        if corner["Label"]:
            ax_speed.text(apex, tel_a["Speed"].max() + 3, corner["Label"],
                          color=col, fontsize=6.5, ha="center", va="bottom",
                          rotation=45)

    # ── Speed traces ──────────────────────────────────────────────────────────
    ax_speed.plot(tel_a["Distance"], tel_a["Speed"],
                  color=_driver_color(driver_a), lw=1.4, label=driver_a, alpha=0.9)
    ax_speed.plot(tel_b["Distance"], tel_b["Speed"],
                  color=_driver_color(driver_b), lw=1.4, label=driver_b, alpha=0.9, ls="--")

    ax_speed.set_ylabel("Speed (km/h)", color="white", fontsize=10)
    ax_speed.legend(framealpha=0.2, labelcolor="white")
    ax_speed.set_title(f"{driver_a} vs {driver_b}  –  {session_label}  |  Silverstone 2025",
                       color="white", fontsize=13, pad=10)
    ax_speed.tick_params(colors="white")
    ax_speed.grid(color="#333", lw=0.5)
    ax_speed.set_xlim(0, max(tel_a["Distance"].max(), tel_b["Distance"].max()))

    # ── Throttle ──────────────────────────────────────────────────────────────
    if "Throttle" in tel_a.columns:
        ax_throttle.plot(tel_a["Distance"], tel_a["Throttle"],
                         color=_driver_color(driver_a), lw=1.0, alpha=0.85)
        ax_throttle.plot(tel_b["Distance"], tel_b["Throttle"],
                         color=_driver_color(driver_b), lw=1.0, alpha=0.85, ls="--")
        ax_throttle.set_ylabel("Throttle %", color="white", fontsize=8)
        ax_throttle.set_ylim(-5, 105)

    # ── Brake ─────────────────────────────────────────────────────────────────
    if "Brake" in tel_a.columns:
        b_a = tel_a["Brake"].clip(0, 100) if tel_a["Brake"].max() > 1 else tel_a["Brake"] * 100
        b_b = tel_b["Brake"].clip(0, 100) if tel_b["Brake"].max() > 1 else tel_b["Brake"] * 100
        ax_brake.plot(tel_a["Distance"], b_a,
                      color=_driver_color(driver_a), lw=1.0, alpha=0.85)
        ax_brake.plot(tel_b["Distance"], b_b,
                      color=_driver_color(driver_b), lw=1.0, alpha=0.85, ls="--")
        ax_brake.set_ylabel("Brake %", color="white", fontsize=8)
        ax_brake.set_ylim(-5, 105)

    for ax in axes:
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(color="#333", lw=0.5)
        ax.spines[:].set_color("#444")

    ax_brake.set_xlabel("Distance (m)", color="white", fontsize=10)

    # ── Corner legend ─────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=v, label=k.capitalize())
               for k, v in CORNER_COLORS.items()]
    ax_speed.legend(handles=ax_speed.get_legend_handles_labels()[0] + patches,
                    labels=ax_speed.get_legend_handles_labels()[1] + [p.get_label() for p in patches],
                    framealpha=0.2, labelcolor="white", fontsize=8, loc="lower right")

    fig.tight_layout(h_pad=0.5)
    if save:
        _save(fig, "01_speed_trace")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2. Corner-type heatmap (who's faster where)
# ─────────────────────────────────────────────────────────────────────────────

def plot_corner_heatmap(
    corner_comparison: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    save: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart showing per-corner min-speed delta,
    grouped and coloured by corner type.
    """
    if corner_comparison.empty:
        log.warning("No corner comparison data – skipping heatmap.")
        return None

    fig, ax = plt.subplots(figsize=(12, max(6, len(corner_comparison) * 0.45)),
                           facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    df = corner_comparison.copy().sort_values("Distance_m")
    labels = [
        f"{row['Label'] or str(int(row['Distance_m'])+'m')}"
        for _, row in df.iterrows()
    ]

    colors = [CORNER_COLORS.get(r["CornerType"], "#888") for _, r in df.iterrows()]
    bars   = ax.barh(labels, df["SpeedDelta"], color=colors, alpha=0.85, edgecolor="#222")

    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.set_xlabel(f"Min-Speed Delta (km/h)  ←  {driver_b} faster  |  {driver_a} faster  →",
                  color="white", fontsize=10)
    ax.set_title(f"Corner Speed Comparison  –  {driver_a} vs {driver_b}  |  Silverstone 2025",
                 color="white", fontsize=13, pad=10)

    ax.tick_params(colors="white", labelsize=8)
    ax.grid(axis="x", color="#333", lw=0.5)
    ax.spines[:].set_color("#444")

    patches = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in CORNER_COLORS.items()]
    ax.legend(handles=patches, framealpha=0.2, labelcolor="white", fontsize=9, loc="lower right")

    fig.tight_layout()
    if save:
        _save(fig, "02_corner_heatmap")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  3. Aero feature radar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_aero_radar(
    sig_a: dict[str, float],
    sig_b: dict[str, float],
    driver_a: str,
    driver_b: str,
    save: bool = True,
) -> plt.Figure:
    """
    Radar (spider) chart comparing 6 normalised aero-proxy features.
    """
    labels = list(sig_a.keys())
    N      = len(labels)

    # Normalise to [0,1] across both drivers
    all_vals = {k: [sig_a[k], sig_b[k]] for k in labels}
    norm_a, norm_b = [], []
    for k in labels:
        lo, hi = min(all_vals[k]), max(all_vals[k])
        rng = hi - lo if hi != lo else 1
        norm_a.append((sig_a[k] - lo) / rng)
        norm_b.append((sig_b[k] - lo) / rng)

    # Circular angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the loop

    norm_a = norm_a + norm_a[:1]
    norm_b = norm_b + norm_b[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"polar": True},
                           facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.plot(angles, norm_a, color=_driver_color(driver_a), lw=2, label=driver_a)
    ax.fill(angles, norm_a, color=_driver_color(driver_a), alpha=0.15)

    ax.plot(angles, norm_b, color=_driver_color(driver_b), lw=2, ls="--", label=driver_b)
    ax.fill(angles, norm_b, color=_driver_color(driver_b), alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, color="white", fontsize=9)
    ax.set_yticklabels([])
    ax.grid(color="#555", lw=0.5)
    ax.spines["polar"].set_color("#555")

    ax.set_title(f"Aero Signature  –  {driver_a} vs {driver_b}  |  Silverstone 2025",
                 color="white", fontsize=13, pad=20)
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=9,
              loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save:
        _save(fig, "03_aero_radar")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  4. Race lap-time evolution + tyre degradation
# ─────────────────────────────────────────────────────────────────────────────

def plot_lap_evolution(
    laps_a: pd.DataFrame,
    laps_b: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    save: bool = True,
) -> plt.Figure:
    """
    Lap-time evolution over the race stint with a trendline (degradation fit).
    """
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    for laps, drv in [(laps_a, driver_a), (laps_b, driver_b)]:
        c = _driver_color(drv)
        ax.scatter(laps["LapNumber"], laps["LapTime_s"],
                   color=c, s=18, alpha=0.7, zorder=3)
        ax.plot(laps["LapNumber"], laps["LapTime_s"],
                color=c, lw=0.8, alpha=0.5)

        # Linear degradation trendline
        if len(laps) > 4:
            z = np.polyfit(laps["LapNumber"], laps["LapTime_s"], 1)
            p = np.poly1d(z)
            x_fit = np.linspace(laps["LapNumber"].min(), laps["LapNumber"].max(), 100)
            ax.plot(x_fit, p(x_fit), color=c, lw=2, ls="--",
                    label=f"{drv}  ({z[0]:+.3f}s/lap)")

    ax.set_xlabel("Lap Number", color="white", fontsize=11)
    ax.set_ylabel("Lap Time (s)", color="white", fontsize=11)
    ax.set_title("Race Lap-Time Evolution & Tyre Degradation  |  Silverstone 2025",
                 color="white", fontsize=13, pad=10)
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(color="#333", lw=0.5)
    ax.spines[:].set_color("#444")

    fig.tight_layout()
    if save:
        _save(fig, "04_lap_evolution")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  5. Predicted vs Actual quali lap time
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_comparison(
    pred_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart comparing predicted vs actual quali lap time per driver,
    with error annotation that tells the aero story.
    """
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    drivers  = pred_df["Driver"].tolist()
    n        = len(drivers)
    x        = np.arange(n)
    w        = 0.35

    for i, row in pred_df.iterrows():
        c = _driver_color(row["Driver"])
        ax.bar(i - w / 2, row["PredictedTime_s"], width=w,
               color=c, alpha=0.5, label=f"{row['Driver']} predicted" if i == 0 else "")
        ax.bar(i + w / 2, row["ActualTime_s"],    width=w,
               color=c, alpha=0.95, label=f"{row['Driver']} actual" if i == 0 else "")

        # Error annotation
        err = row["Error_s"]
        err_str = f"{err:+.3f}s"
        y_top = max(row["PredictedTime_s"], row["ActualTime_s"]) + 0.15
        ax.annotate(err_str, xy=(i, y_top), ha="center", color="white",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['Driver']}\n{r['Team']}" for _, r in pred_df.iterrows()],
        color="white", fontsize=10
    )

    # Format y-axis as M:SS.mmm
    ylim_lo = pred_df[["PredictedTime_s", "ActualTime_s"]].min().min() - 0.5
    ylim_hi = pred_df[["PredictedTime_s", "ActualTime_s"]].max().max() + 1.0
    ticks   = np.arange(np.floor(ylim_lo), np.ceil(ylim_hi), 0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_fmt_laptime(t) for t in ticks], color="white", fontsize=8)
    ax.set_ylim(ylim_lo, ylim_hi)

    ax.set_title("Predicted vs Actual Quali Lap Time  –  Aero Setup Effect  |  Silverstone 2025",
                 color="white", fontsize=12, pad=10)
    ax.set_ylabel("Lap Time", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.grid(axis="y", color="#333", lw=0.5)
    ax.spines[:].set_color("#444")

    # Legend: predicted = faded, actual = solid
    pred_patch  = mpatches.Patch(color="#888888", alpha=0.5,  label="Predicted (from race laps)")
    act_patch   = mpatches.Patch(color="#888888", alpha=0.95, label="Actual Quali Time")
    ax.legend(handles=[pred_patch, act_patch], framealpha=0.2,
              labelcolor="white", fontsize=9)

    fig.tight_layout()
    if save:
        _save(fig, "05_prediction_vs_actual")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  6. Feature importance from XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart of top-N XGBoost feature importances.
    """
    df  = importance_df.head(top_n).iloc[::-1]   # reverse for bottom-to-top

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    bars = ax.barh(df["Feature"], df["Importance"],
                   color="#3671C6", alpha=0.85, edgecolor="#222")

    ax.set_xlabel("Feature Importance (Gain)", color="white", fontsize=10)
    ax.set_title("XGBoost Feature Importance  –  Lap-Time Predictor  |  Silverstone 2025",
                 color="white", fontsize=12, pad=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(axis="x", color="#333", lw=0.5)
    ax.spines[:].set_color("#444")

    fig.tight_layout()
    if save:
        _save(fig, "06_feature_importance")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  7. Delta-time chart (gap evolution over one lap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_time(
    delta_df: pd.DataFrame,
    corners_a: pd.DataFrame,
    driver_a: str,
    driver_b: str,
    session_label: str = "Fastest Race Lap",
    save: bool = True,
) -> plt.Figure:
    """
    Gap chart showing cumulative time delta between two drivers over one lap.
    Positive = driver_a ahead; negative = driver_b ahead.
    """
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # Corner shading
    for _, corner in corners_a.iterrows():
        col  = CORNER_COLORS.get(corner["CornerType"], "#888")
        apex = corner["Distance_m"]
        ax.axvspan(apex - 60, apex + 60, color=col, alpha=0.07, lw=0)

    ax.fill_between(delta_df["Distance"], delta_df["Delta_s"],
                    where=delta_df["Delta_s"] >= 0,
                    color=_driver_color(driver_a), alpha=0.5, interpolate=True)
    ax.fill_between(delta_df["Distance"], delta_df["Delta_s"],
                    where=delta_df["Delta_s"] < 0,
                    color=_driver_color(driver_b), alpha=0.5, interpolate=True)

    ax.plot(delta_df["Distance"], delta_df["Delta_s"],
            color="white", lw=1.2, alpha=0.9)
    ax.axhline(0, color="#888", lw=0.8, ls="--")

    ax.set_xlabel("Distance (m)", color="white", fontsize=11)
    ax.set_ylabel(f"Δ Time (s)  [{driver_a} ahead ↑ / {driver_b} ahead ↓]",
                  color="white", fontsize=10)
    ax.set_title(f"Lap-Delta  –  {driver_a} vs {driver_b}  |  {session_label}  |  Silverstone 2025",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.grid(color="#333", lw=0.5)
    ax.spines[:].set_color("#444")

    fig.tight_layout()
    if save:
        _save(fig, "07_delta_time")
    return fig
