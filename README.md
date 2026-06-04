# F1 Aero Analysis — Silverstone 2025

**Comparing McLaren vs Red Bull aero setups through telemetry, corner analysis and lap-time prediction.**

The hypothesis: McLaren ran a higher-downforce setup at Silverstone 2025 while Red Bull ran low-drag. This manifests in the data as:
- **RB advantage** → higher top speed on Hangar Straight
- **MCL advantage** → higher minimum corner speeds at Copse, Maggotts-Becketts (fast corners)

An XGBoost model is trained on race-lap telemetry features and used to predict qualifying lap times. The prediction error per driver is the clean signal of how much their aero setup amplified performance in qualifying vs race conditions.

---

## Project Structure

```
f1-aero-analysis/
├── config.py                   # All constants (year, drivers, thresholds)
├── main.py                     # Full pipeline orchestrator
├── requirements.txt
├── src/
│   ├── pipeline/
│   │   ├── loader.py           # FastF1 session loader + cache management
│   │   ├── telemetry.py        # Telemetry cleaning, derived channels, alignment
│   │   └── corners.py          # Corner detection, classification, comparison
│   ├── features/
│   │   ├── aero.py             # Aero-proxy feature extraction (11 feature groups)
│   │   └── sector.py           # Sector + mini-sector features
│   ├── models/
│   │   └── laptime.py          # XGBoost lap-time predictor
│   └── viz/
│       └── plots.py            # 7 matplotlib visualizations
├── data/cache/                 # FastF1 disk cache (auto-populated)
└── outputs/                    # Saved plots (auto-populated)
```

---

## Outputs

| # | File | Description |
|---|------|-------------|
| 1 | `01_speed_trace.png` | VER vs NOR speed/throttle/brake overlaid, corner shading |
| 2 | `02_corner_heatmap.png` | Per-corner min-speed delta, grouped by corner type |
| 3 | `03_aero_radar.png` | 6-feature aero fingerprint radar chart |
| 4 | `04_lap_evolution.png` | Race lap times with degradation trendlines |
| 5 | `05_prediction_vs_actual.png` | Model prediction vs actual quali time (aero effect) |
| 6 | `06_feature_importance.png` | XGBoost feature importances (gain) |
| 7 | `07_delta_time.png` | Cumulative lap-delta across one lap |

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The first run will download session data and populate `data/cache/`. Subsequent runs are fast.

---

## Aero Features (XGBoost inputs)

| Feature group | Columns | Aero insight |
|---|---|---|
| Top speed | `top_speed`, `hangar_straight_top_speed` | Low drag → high straight speed |
| Corner min speed | `min_speed_{slow/medium/fast}_mean` | High DF → high min corner speed |
| DRS gain | `drs_speed_delta`, `drs_open_fraction` | Low drag → bigger DRS delta |
| Braking distance | `braking_dist_{mean/slow/fast}` | High DF → shorter braking distances |
| Exit acceleration | `exit_accel_{mean/fast/slow}` | Mechanical + aero traction |
| Lateral G | `lateral_g_fast_mean` | High DF → higher lateral loading |
| Throttle application | `throttle_app_dist_{mean/slow}` | Grip confidence out of corners |
| Speed consistency | `speed_std_{medium/fast}` | Stability = high DF |
| Sector speed | `S1/S2/S3_mean_speed`, `_max_speed` | Zone-level performance |
| Engine | `mean_rpm`, `max_rpm` | Power delivery |
| Throttle trace | `full_throttle_pct`, `mean_throttle` | Aero balance on straights |

---

## Model Design

- **Train**: XGBoost on all valid race laps (both drivers, filtered for SC/pit outliers)
- **Predict**: Quali lap time from quali telemetry features
- **Signal**: `predicted − actual` per driver reveals which car's setup unlocked more pace in qualifying
  - Over-prediction → driver found more pace in quali than race patterns suggest (aero optimisation)
  - Under-prediction → race setup was more conservative than quali required
