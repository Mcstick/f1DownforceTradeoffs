# F1 Aero Analysis — Silverstone 2025

**Comparing McLaren vs Red Bull aero setups through telemetry, corner analysis and lap-time prediction.**

The hypothesis: McLaren ran a higher-downforce setup at Silverstone 2025 while Red Bull ran low-drag. This manifests in the data as:
- **RB advantage** → higher top speed on Hangar Straight
- **MCL advantage** → higher minimum corner speeds at Copse, Maggotts-Becketts (fast corners)

An XGBoost model is trained on race-lap telemetry features and used to predict qualifying lap times. The prediction error per driver is the clean signal of how much their aero setup amplified performance in qualifying vs race conditions.

---

## Results & Insights

### Quali Lap Time Prediction

| Driver | Team | Predicted | Actual | Error |
|--------|------|-----------|--------|-------|
| VER | Red Bull Racing | ~1:25.9 | ~1:24.8 | +1.083s |
| NOR | McLaren | ~1:25.9 | ~1:24.9 | +0.964s |

**The key number is the delta: 0.119s.**

Norris's prediction error is smaller than Verstappen's, meaning McLaren's Q3 pace was more consistent with their race telemetry patterns than Red Bull's was. In plain terms — McLaren's high-downforce setup translated more predictably from race to qualifying conditions.

Red Bull found an extra ~0.12s in qualifying that their race-lap data couldn't account for. This is consistent with a low-drag setup that comes alive on a single hot lap with lower fuel load and maximum engine modes — exactly the kind of gain you'd expect from a setup optimised for peak straight-line speed over a single lap rather than race-long stability.

### What this confirms about the aero tradeoff

- **McLaren's high-downforce setup** produced race-pace telemetry (corner speeds, braking distances, lateral G) that closely mirrored their qualifying behaviour. The car was planted and consistent across both conditions.
- **Red Bull's low-drag setup** showed a larger gap between race and qualifying pace. The setup unlocked additional performance in qualifying that didn't show up in race laps — a classic signature of a car that rewards low-fuel, single-lap conditions more than tyre-limited race stints.
- The model's ~1s absolute error on both drivers (trained on ~50 laps of data) is a strong result. The relative difference between the two errors is the aero signal.

### Methodology note

The model is trained on race laps + Q1/Q2 qualifying laps, with each driver's fastest Q3 lap held out as the prediction target. This prevents data leakage while giving the model enough exposure to qualifying-pace telemetry to bridge the ~8s gap between race and quali conditions caused by fuel load, tyre modes, and engine deployment differences.

---

## Project Structure

```
f1DownforceTradeoffs/
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
pip3 install -r requirements.txt --break-system-packages
python3 main.py
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

- **Train**: XGBoost on all valid race laps + Q1/Q2 laps (both drivers, filtered for SC/pit outliers)
- **Predict**: Q3 fastest lap time from qualifying telemetry features (held out from training)
- **Signal**: `predicted − actual` per driver reveals which car's setup unlocked more pace in qualifying
  - Smaller error → setup translated consistently from race to quali conditions (high downforce)
  - Larger error → setup found extra pace in quali that race laps couldn't predict (low drag)