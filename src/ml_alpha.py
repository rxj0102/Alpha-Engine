"""
ml_alpha.py — Stacked ensemble for cross-sectional return forecasting.

Architecture (two-stage [ML-1])
---------------------------------
Stage 1  Walk-forward OOF cross-validation
         Base learners : RandomForest, GradientBoosting, SVR, Ridge
         Meta-learner  : Ridge trained on OOF base predictions
         → Unbiased IC / R² / calibration metrics

Stage 2  Full refit on same training data
         All base models + meta-learner refitted for deployment
         → Calibration-consistent inference (avoids retraining bias)

Alpha metric
-----------
Information Coefficient (IC) = Spearman rank correlation between
predicted and realised forward returns. IC > 0 → valid view.
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf  # noqa: F401 (re-exported for convenience)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import MLConfig, SEED

log = logging.getLogger(__name__)


class MLAlphaEngine:
    """
    Stacked ensemble (RF + GBM + SVR + Ridge) for cross-sectional alpha generation.

    Parameters
    ----------
    cfg : MLConfig
        Hyper-parameters for base learners and cross-validation.
    """

    def __init__(self, cfg: MLConfig = None) -> None:
        self.cfg = cfg or MLConfig()
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.feat_importance: Dict = {}

    # ── Data preparation ───────────────────────────────────────────────────

    def prepare_data(self, features: pd.DataFrame, horizon: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix X and forward-return target y.

        Uses compound forward return p(t+h)/p(t) - 1 to avoid sign-flip
        artefacts from composing multiple pct_change() values.

        Parameters
        ----------
        features : pd.DataFrame
            Output of DataEngine.engineer_features() for one ticker.
        horizon : int, optional
            Prediction horizon in trading days; defaults to cfg.horizon.

        Returns
        -------
        X : pd.DataFrame, y : pd.Series
            Aligned, NaN-free feature/label pairs.
        """
        h = horizon or self.cfg.horizon
        p = features["price"]
        y = (p.shift(-h) / p - 1).rename("fwd_return")
        X = features.drop(columns=["return_1d", "price"], errors="ignore")
        valid = X.notna().all(axis=1) & y.notna()
        return X[valid], y[valid]

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _ic(actuals: np.ndarray, preds: np.ndarray) -> float:
        """Spearman IC — industry-standard alpha quality metric."""
        return float(stats.spearmanr(actuals, preds).statistic)

    def _build_base_models(self):
        """Instantiate base learners from config."""
        cfg = self.cfg
        return [
            ("rf", RandomForestRegressor(
                n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                min_samples_split=cfg.rf_min_samples_split, max_features=cfg.rf_max_features,
                random_state=SEED, n_jobs=-1,
            )),
            ("gbr", GradientBoostingRegressor(
                n_estimators=cfg.gbr_n_estimators, max_depth=cfg.gbr_max_depth,
                learning_rate=cfg.gbr_learning_rate, subsample=cfg.gbr_subsample,
                random_state=SEED,
            )),
            ("svr", SVR(kernel="rbf", C=cfg.svr_c, epsilon=cfg.svr_epsilon, gamma="scale")),
            ("ridge", Ridge(alpha=cfg.ridge_alpha)),
        ]

    # ── Training ───────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series, ticker: str) -> Dict:
        """
        Two-stage walk-forward training.

        Stage 1: TimeSeriesSplit OOF → unbiased IC / R² / calibration slope.
        Stage 2: Full refit on all data → production-ready models stored in self.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (no NaNs).
        y : pd.Series
            Forward return targets.
        ticker : str
            Asset identifier used as the model store key.

        Returns
        -------
        dict
            {"IC": float, "r2": float, "calibration_slope": float}
        """
        n = len(X)
        tscv = TimeSeriesSplit(n_splits=self.cfg.n_cv_splits)
        base_models = self._build_base_models()

        oof = {name: np.full(n, np.nan) for name, _ in base_models}
        actuals = np.full(n, np.nan)

        # Stage 1: OOF walk-forward CV (per-fold scaling to prevent leakage)
        for train_idx, test_idx in tscv.split(X):
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(X.iloc[train_idx])
            Xte_s = scaler.transform(X.iloc[test_idx])
            for name, m in base_models:
                m.fit(Xtr_s, y.iloc[train_idx])
                oof[name][test_idx] = m.predict(Xte_s)
            actuals[test_idx] = y.iloc[test_idx].values

        valid = ~np.isnan(actuals)
        meta_X_oof = np.column_stack([oof[name][valid] for name, _ in base_models])
        meta_y_oof = actuals[valid]

        meta_scaler_oof = StandardScaler()
        meta_oof = Ridge(alpha=self.cfg.meta_ridge_alpha)
        meta_oof.fit(meta_scaler_oof.fit_transform(meta_X_oof), meta_y_oof)
        oof_preds = meta_oof.predict(meta_scaler_oof.transform(meta_X_oof))

        oof_ic = self._ic(meta_y_oof, oof_preds)
        oof_r2 = r2_score(meta_y_oof, oof_preds)
        cal_slope = float(np.polyfit(meta_y_oof, oof_preds, 1)[0])

        # Stage 2: Full refit — base models and meta-learner on same data [ML-1]
        final_scaler = StandardScaler()
        Xall_s = final_scaler.fit_transform(X)
        for name, m in base_models:
            m.fit(Xall_s, y)

        full_base = np.column_stack([m.predict(Xall_s) for _, m in base_models])
        meta_scaler_final = StandardScaler()
        meta_final = Ridge(alpha=self.cfg.meta_ridge_alpha)
        meta_final.fit(meta_scaler_final.fit_transform(full_base), y.values)

        self.models[ticker] = {name: m for name, m in base_models}
        self.models[ticker].update({"meta": meta_final, "meta_scaler": meta_scaler_final})
        self.scalers[ticker] = final_scaler
        self.feat_importance[ticker] = pd.Series(
            dict(zip(X.columns, base_models[0][1].feature_importances_))
        ).sort_values(ascending=False)

        log.info("  %s: OOF IC=%.4f  R²=%.4f  CalSlope=%.3f", ticker, oof_ic, oof_r2, cal_slope)
        return {"IC": oof_ic, "r2": oof_r2, "calibration_slope": cal_slope}

    # ── Inference ──────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame, ticker: str) -> np.ndarray:
        """
        Generate return forecast for new observations.

        Parameters
        ----------
        X : pd.DataFrame
            Feature rows (must match training columns).
        ticker : str
            Must have been previously trained via `train()`.

        Returns
        -------
        np.ndarray
            Predicted forward returns (shape: [n_rows]).
        """
        m = self.models[ticker]
        Xs = self.scalers[ticker].transform(X)
        base = np.column_stack([m[name].predict(Xs) for name in ["rf", "gbr", "svr", "ridge"]])
        return m["meta"].predict(m["meta_scaler"].transform(base))
