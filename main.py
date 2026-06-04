# ─────────────────────────────────────────────
#  main.py  –  Full pipeline orchestrator
#
#  Run:  python main.py
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Add src to path so imports resolve cleanly ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for all environments
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from src.utils import setup_logging, ensure_dirs
from config   import OUTPUT_DIR, CACHE_DIR, DRIVER_CODES, DRIVERS

setup_logging()
log = logging.getLogger("main")

from src.pipeline.loader   import SessionBundle
from src.pipeline.telemetry import (
    clean_telemetry, add_derived_channels,
    align_by_distance, compute_delta_time,
    filter_valid_race_laps,
)
from src.pipeline.corners  import detect_corners, corner_stats, compare_corner_stats
from src.features.aero     import extract_aero_features, build_feature_matrix, aero_signature, RADAR_FEATURES
from src.features.sector   import sector_features
from src.models.laptime    import LapTimeModel
from src.viz.plots         import (
    plot_speed_trace, plot_corner_heatmap, plot_aero_radar,
    plot_lap_evolution, plot_prediction_comparison,
    plot_feature_importance, plot_delta_time,
)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs(OUTPUT_DIR, CACHE_DIR, "data")

    # ── Stage 1: Load sessions ────────────────────────────────────────────────
    log.info("═══════════════  STAGE 1 – LOADING  ═══════════════")
    bundle = SessionBundle().load()

    # ── Stage 2: Process telemetry for fastest laps ───────────────────────────
    log.info("═══════════════  STAGE 2 – TELEMETRY  ══════════════")

    tel_race, tel_quali = {}, {}

    for drv in DRIVER_CODES:
        for session in ["race", "quali"]:
            raw = bundle.get_telemetry(drv, session)
            tel = clean_telemetry(raw)
            tel = add_derived_channels(tel)
            if session == "race":
                tel_race[drv] = tel
            else:
                tel_quali[drv] = tel
            log.info("%s %-5s  –  %d telemetry rows", drv, session, len(tel))

    # ── Stage 3: Corner classification ───────────────────────────────────────
    log.info("═══════════════  STAGE 3 – CORNERS  ════════════════")

    drv_a, drv_b = DRIVER_CODES[0], DRIVER_CODES[1]

    corners_race = {}
    cstats_race  = {}
    for drv in DRIVER_CODES:
        corners_race[drv] = detect_corners(tel_race[drv])
        cstats_race[drv]  = corner_stats(tel_race[drv], corners_race[drv])
        log.info("%s  –  %d corners detected  (slow=%d, med=%d, fast=%d)",
                 drv,
                 len(corners_race[drv]),
                 (corners_race[drv]["CornerType"] == "slow").sum(),
                 (corners_race[drv]["CornerType"] == "medium").sum(),
                 (corners_race[drv]["CornerType"] == "fast").sum(),
                 )

    corner_cmp = compare_corner_stats(cstats_race[drv_a], cstats_race[drv_b], drv_a, drv_b)

    # ── Stage 4: Feature engineering ─────────────────────────────────────────
    log.info("═══════════════  STAGE 4 – FEATURES  ═══════════════")

    # Build race-lap feature matrix (all valid laps, both drivers)
    race_feature_rows = []
    for drv in DRIVER_CODES:
        valid_laps = filter_valid_race_laps(bundle.race_laps[drv])
        log.info("%s  –  %d valid race laps for feature extraction", drv, len(valid_laps))

        for _, lap_row in valid_laps.iterrows():
            try:
                lap_obj = bundle.race_session.laps.pick_drivers(drv).loc[lap_row.name]
                tel     = clean_telemetry(lap_obj.get_car_data().add_distance().merge_channels(lap_obj.get_pos_data()))
                tel     = add_derived_channels(tel)
                crns    = detect_corners(tel)
                cst     = corner_stats(tel, crns)

                feats = extract_aero_features(
                    tel, cst,
                    lap_time_s=lap_row["LapTime_s"],
                    driver=drv,
                    lap_number=int(lap_row.get("LapNumber", 0)),
                )
                feats.update(sector_features(tel))
                race_feature_rows.append(feats)

            except Exception as exc:
                log.debug("Skipping lap %s/%s: %s", drv, lap_row.get("LapNumber"), exc)

    race_features = build_feature_matrix(race_feature_rows)
    log.info("Race feature matrix: %s", race_features.shape)

    # Build SINGLE-ROW quali feature matrix per driver (fastest quali lap)
    quali_feature_rows = []
    actual_quali_times = {}
    for drv in DRIVER_CODES:
        try:
            crns = detect_corners(tel_quali[drv])
            cst  = corner_stats(tel_quali[drv], crns)
            feats = extract_aero_features(tel_quali[drv], cst, driver=drv)
            feats.update(sector_features(tel_quali[drv]))
            quali_feature_rows.append(feats)
            actual_quali_times[drv] = bundle.quali_best[drv]["LapTime"].total_seconds()
        except Exception as exc:
            log.warning("Quali feature extraction failed for %s: %s", drv, exc)

    quali_features = build_feature_matrix(quali_feature_rows)

    # ── Stage 5: ML model ─────────────────────────────────────────────────────
    log.info("═══════════════  STAGE 5 – MODEL  ══════════════════")

    model    = LapTimeModel().fit(race_features)
    pred_df  = model.predict_quali_and_compare(quali_features, actual_quali_times)
    imp_df   = model.feature_importance()

    log.info("\n%s", pred_df[["Driver", "Team", "PredictedTime_s", "ActualTime_s",
                               "Error_s", "Interpretation"]].to_string(index=False))

    # ── Stage 6: Visualizations ───────────────────────────────────────────────
    log.info("═══════════════  STAGE 6 – PLOTS  ══════════════════")

    # Align telemetry for delta / overlay plots
    tel_a_aligned, tel_b_aligned = align_by_distance(tel_race[drv_a], tel_race[drv_b])
    delta_df = compute_delta_time(tel_a_aligned, tel_b_aligned)

    # 1. Speed trace
    plot_speed_trace(tel_race[drv_a], tel_race[drv_b],
                     corners_race[drv_a], drv_a, drv_b,
                     session_label="Fastest Race Lap")

    # 2. Corner heatmap
    plot_corner_heatmap(corner_cmp, drv_a, drv_b)

    # 3. Aero radar – use *race* best lap features
    race_feat_a = race_features[race_features["driver"] == drv_a].iloc[0] \
                  if not race_features[race_features["driver"] == drv_a].empty else None
    race_feat_b = race_features[race_features["driver"] == drv_b].iloc[0] \
                  if not race_features[race_features["driver"] == drv_b].empty else None

    if race_feat_a is not None and race_feat_b is not None:
        sig_a = aero_signature(race_feat_a)
        sig_b = aero_signature(race_feat_b)
        # Drop NaN-valued features from radar
        valid_keys = [k for k in sig_a if not (isinstance(sig_a[k], float) and
                      (sig_a[k] != sig_a[k]) or isinstance(sig_b[k], float) and (sig_b[k] != sig_b[k]))]
        plot_aero_radar(
            {k: sig_a[k] for k in valid_keys},
            {k: sig_b[k] for k in valid_keys},
            drv_a, drv_b,
        )

    # 4. Lap evolution
    plot_lap_evolution(
        filter_valid_race_laps(bundle.race_laps[drv_a]),
        filter_valid_race_laps(bundle.race_laps[drv_b]),
        drv_a, drv_b,
    )

    # 5. Prediction vs actual
    plot_prediction_comparison(pred_df)

    # 6. Feature importance
    plot_feature_importance(imp_df)

    # 7. Delta time
    plot_delta_time(delta_df, corners_race[drv_a], drv_a, drv_b)

    log.info("═══════════════  DONE  ══════════════════════════════")
    log.info("All outputs saved to  ./%s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
