# ─────────────────────────────────────────────
#  src/models/laptime.py
#  XGBoost lap-time predictor:
#  Train on race laps → predict quali lap time → compare vs actual.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config import XGBOOST_PARAMS, TEST_SIZE, MODEL_RANDOM_SEED, DRIVERS

log = logging.getLogger(__name__)

# Feature columns used by the model (must exist in the feature matrix)
# Metadata / label columns are excluded automatically.
LABEL_COL   = "lap_time_s"
META_COLS   = {"driver", "lap_number", "session"}   # never used as features


# ─────────────────────────────────────────────────────────────────────────────
#  LapTimeModel
# ─────────────────────────────────────────────────────────────────────────────

class LapTimeModel:
    """
    XGBoost-based lap-time predictor trained on race-lap features.

    Workflow
    --------
    1. ``fit(race_features)``         – train + cross-validate on race laps
    2. ``predict(features)``          – predict lap time(s) for any feature df
    3. ``predict_quali_and_compare``  – convenience: predict quali laps and
                                        compare to actual times per driver
    """

    def __init__(self):
        self.model: XGBRegressor | None = None
        self.feature_cols: list[str]    = []
        self.train_metrics: dict        = {}
        self.is_fitted: bool            = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, race_features: pd.DataFrame) -> "LapTimeModel":
        """
        Train XGBoost on the race-lap feature matrix.

        Parameters
        ----------
        race_features : DataFrame produced by ``build_feature_matrix``
                        Must contain a ``lap_time_s`` column.
        """
        df = race_features.dropna(subset=[LABEL_COL]).copy()

        # Identify feature columns (numeric, not meta, not label)
        self.feature_cols = [
            c for c in df.columns
            if c not in META_COLS and c != LABEL_COL
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        X = df[self.feature_cols].copy()
        y = df[LABEL_COL].values

        # Median-impute remaining NaNs
        for col in self.feature_cols:
            X[col] = X[col].fillna(X[col].median())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=MODEL_RANDOM_SEED
        )

        self.model = XGBRegressor(**XGBOOST_PARAMS)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        y_pred_test = self.model.predict(X_test)
        cv_scores   = cross_val_score(
            XGBRegressor(**XGBOOST_PARAMS), X, y,
            scoring="neg_mean_absolute_error", cv=5
        )

        self.train_metrics = {
            "mae_test":   mean_absolute_error(y_test, y_pred_test),
            "r2_test":    r2_score(y_test, y_pred_test),
            "cv_mae_mean": -cv_scores.mean(),
            "cv_mae_std":   cv_scores.std(),
            "n_train":     len(X_train),
            "n_test":      len(X_test),
        }
        self.is_fitted = True

        log.info(
            "Model trained | MAE=%.3fs | R²=%.4f | CV-MAE=%.3f±%.3f",
            self.train_metrics["mae_test"],
            self.train_metrics["r2_test"],
            self.train_metrics["cv_mae_mean"],
            self.train_metrics["cv_mae_std"],
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict lap time(s) for a feature DataFrame."""
        assert self.is_fitted, "Call .fit() before .predict()"

        X = features[self.feature_cols].copy()
        for col in self.feature_cols:
            X[col] = X[col].fillna(X[col].median())

        return self.model.predict(X)

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances (gain-based)."""
        assert self.is_fitted
        imp = self.model.get_booster().get_score(importance_type="gain")
        df  = pd.DataFrame({"Feature": list(imp.keys()), "Importance": list(imp.values())})
        return df.sort_values("Importance", ascending=False).reset_index(drop=True)

    # ── Quali prediction & comparison ─────────────────────────────────────────

    def predict_quali_and_compare(
        self,
        quali_features: pd.DataFrame,
        actual_quali_times: dict[str, float],
    ) -> pd.DataFrame:
        """
        Predict the quali lap time for each driver from their quali-session
        telemetry features, then compare with the actual recorded time.

        Parameters
        ----------
        quali_features      : feature matrix for quali laps (one row per driver)
        actual_quali_times  : dict mapping driver_code → actual lap time in seconds

        Returns
        -------
        DataFrame with columns:
            Driver, PredictedTime_s, ActualTime_s, Error_s, Error_pct,
            Interpretation
        """
        assert self.is_fitted

        rows = []
        for _, row in quali_features.iterrows():
            drv    = row.get("driver", "?")
            actual = actual_quali_times.get(drv, np.nan)
            pred   = float(self.predict(pd.DataFrame([row]))[0])
            error  = pred - actual
            pct    = (error / actual) * 100 if actual else np.nan

            # Interpretation:
            # If the model (trained on race data) over-predicts quali time,
            # it means the driver extracted *more* from the car in quali than
            # the race-lap pattern suggests → aero setup optimised for quali
            interpretation = (
                f"Model over-predicts by {abs(error):.3f}s – {drv} found more pace in quali"
                if error > 0
                else f"Model under-predicts by {abs(error):.3f}s – {drv} race setup more conservative"
            )

            rows.append({
                "Driver":          drv,
                "FullName":        DRIVERS.get(drv, {}).get("full_name", drv),
                "Team":            DRIVERS.get(drv, {}).get("team", ""),
                "PredictedTime_s": pred,
                "ActualTime_s":    actual,
                "Error_s":         error,
                "Error_pct":       pct,
                "Interpretation":  interpretation,
            })

        return pd.DataFrame(rows)
