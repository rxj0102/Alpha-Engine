"""
tests/test_ml_alpha.py — Unit tests for MLAlphaEngine.

Tests cover:
- prepare_data: target construction, NaN alignment
- _ic: Spearman IC edge cases
- train: output keys, IC range, model storage
- predict: output shape, reproducibility
"""

import numpy as np
import pandas as pd
import pytest

from src.ml_alpha import MLAlphaEngine
from src.config import MLConfig


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_features():
    """300-row feature DataFrame with 'price' and a few numeric factors."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    price = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.01, n))
    df = pd.DataFrame(
        {
            "price": price,
            "return_1d": np.random.normal(0, 0.01, n),
            "mom_5d": np.random.normal(0, 0.05, n),
            "mom_21d": np.random.normal(0, 0.08, n),
            "vol_21d": np.abs(np.random.normal(0.15, 0.03, n)),
            "rsi": np.random.uniform(30, 70, n),
            "macd": np.random.normal(0, 0.5, n),
        },
        index=dates,
    )
    return df


@pytest.fixture
def small_cfg():
    """Minimal MLConfig for fast testing."""
    return MLConfig(
        rf_n_estimators=10,
        gbr_n_estimators=10,
        n_cv_splits=3,
        horizon=5,
    )


# ── Tests: prepare_data ────────────────────────────────────────────────────────

class TestPrepareData:
    def test_removes_price_and_return1d(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features, horizon=5)
        assert "price" not in X.columns
        assert "return_1d" not in X.columns

    def test_no_nans_in_output(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features, horizon=5)
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_x_y_aligned(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features, horizon=5)
        assert len(X) == len(y)
        assert (X.index == y.index).all()

    def test_target_is_compound_return(self, synthetic_features, small_cfg):
        """Target = p(t+h)/p(t) - 1, not pct_change."""
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features, horizon=1)
        p = synthetic_features["price"]
        expected = (p.shift(-1) / p - 1).dropna()
        overlap = y.index.intersection(expected.index)
        np.testing.assert_allclose(y[overlap].values, expected[overlap].values, rtol=1e-10)


# ── Tests: _ic ─────────────────────────────────────────────────────────────────

class TestIC:
    def test_perfect_positive_correlation(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        assert MLAlphaEngine._ic(a, a) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([5, 4, 3, 2, 1], dtype=float)
        assert MLAlphaEngine._ic(a, b) == pytest.approx(-1.0)

    def test_ic_bounded(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=100)
        b = rng.normal(size=100)
        ic = MLAlphaEngine._ic(a, b)
        assert -1.0 <= ic <= 1.0


# ── Tests: train ───────────────────────────────────────────────────────────────

class TestTrain:
    def test_returns_required_keys(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features)
        result = eng.train(X, y, "SPY")
        assert "IC" in result
        assert "r2" in result
        assert "calibration_slope" in result

    def test_ic_in_valid_range(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features)
        result = eng.train(X, y, "SPY")
        assert -1.0 <= result["IC"] <= 1.0

    def test_models_stored(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features)
        eng.train(X, y, "SPY")
        assert "SPY" in eng.models
        for name in ["rf", "gbr", "svr", "ridge", "meta", "meta_scaler"]:
            assert name in eng.models["SPY"], f"Missing model component: {name}"

    def test_scaler_stored(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features)
        eng.train(X, y, "SPY")
        assert "SPY" in eng.scalers


# ── Tests: predict ─────────────────────────────────────────────────────────────

class TestPredict:
    def test_output_shape(self, synthetic_features, small_cfg):
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features)
        eng.train(X, y, "SPY")
        preds = eng.predict(X.iloc[-10:], "SPY")
        assert preds.shape == (10,)

    def test_reproducible(self, synthetic_features, small_cfg):
        """Same input must produce same predictions."""
        eng = MLAlphaEngine(cfg=small_cfg)
        X, y = eng.prepare_data(synthetic_features)
        eng.train(X, y, "SPY")
        p1 = eng.predict(X.iloc[-5:], "SPY")
        p2 = eng.predict(X.iloc[-5:], "SPY")
        np.testing.assert_array_equal(p1, p2)
