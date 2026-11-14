"""
tests/test_data.py — Unit tests for DataEngine.

Tests cover:
- open data helpers: par bond returns, equal BL prior
- engineer_features: output shape, no look-ahead, feature completeness
- ffill gap-filling behaviour
"""

import numpy as np
import pandas as pd
import pytest

from src.data import DataEngine


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def price_df():
    """Synthetic daily price series for 3 tickers over 300 days."""
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    np.random.seed(42)
    prices = pd.DataFrame(
        {
            "SPY": 400 * np.cumprod(1 + np.random.normal(0.0004, 0.01, 300)),
            "TLT": 120 * np.cumprod(1 + np.random.normal(-0.0001, 0.005, 300)),
            "GLD": 180 * np.cumprod(1 + np.random.normal(0.0002, 0.007, 300)),
        },
        index=dates,
    )
    return prices


@pytest.fixture
def data_engine(price_df):
    """DataEngine pre-loaded with synthetic prices."""
    de = DataEngine("2022-01-01", "2023-01-01")
    de.data["prices"] = price_df
    return de


# ── Tests: open data helpers ───────────────────────────────────────────────────

class TestOpenData:
    def test_par_bond_returns_sign(self):
        from src.opendata import par_bond_returns
        y = pd.Series([4.0, 4.0, 4.5, 4.0], index=pd.date_range("2024-01-01", periods=4))
        r = par_bond_returns(y, 10)
        assert r.iloc[1] < 0 < r.iloc[2]          # yield up, price down
        assert abs(r.iloc[0] - 0.04 / 252) < 1e-12  # flat yield earns carry

    def test_equal_prior_weights(self):
        caps = DataEngine("2020-01-01", "2020-12-31").fetch_market_caps(["SPY", "TLT"])
        assert (caps == 1.0).all()


# ── Tests: engineer_features ───────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_returns_dict_with_all_tickers(self, data_engine, price_df):
        feat = data_engine.engineer_features()
        assert set(feat.keys()) == set(price_df.columns)

    def test_feature_columns_present(self, data_engine):
        feat = data_engine.engineer_features()
        f = feat["SPY"]
        expected = [
            "mom_5d", "mom_21d", "mom_63d", "mom_252d",
            "vol_5d", "vol_21d", "vol_63d", "vol_252d",
            "price_to_sma20", "sma20_to_sma50",
            "macd", "macd_hist", "bb_width", "rsi",
            "ret_mean_63d", "ret_skew_63d", "ret_kurt_63d",
        ]
        for col in expected:
            assert col in f.columns, f"Missing feature: {col}"

    def test_cutoff_date_prevents_lookahead(self, data_engine, price_df):
        """Features with cutoff_date must not use data beyond that date."""
        cutoff = "2022-06-01"
        feat = data_engine.engineer_features(cutoff_date=cutoff)
        for ticker, df in feat.items():
            assert df.index.max() <= pd.Timestamp(cutoff), \
                f"{ticker}: feature index exceeds cutoff_date"

    def test_no_lookahead_shorter_than_full(self, data_engine, price_df):
        """Features up to cutoff should have fewer rows than full-period features."""
        full_feat = data_engine.engineer_features()
        cutoff_feat = data_engine.engineer_features(cutoff_date="2022-06-01")
        assert len(cutoff_feat["SPY"]) < len(full_feat["SPY"])

    def test_feature_values_are_finite_after_warmup(self, data_engine):
        """Most features need ≥252 days to stabilise — check past warmup period."""
        feat = data_engine.engineer_features()
        f = feat["SPY"].iloc[252:]
        finite_ratio = f.drop(columns=["price"]).notna().mean().mean()
        assert finite_ratio > 0.95, f"Too many NaNs after warmup: {finite_ratio:.2%}"


# ── Tests: gap-filling ─────────────────────────────────────────────────────────

class TestGapFilling:
    def test_ffill_short_gaps(self):
        """ffill(limit=5) should fill up to 5 consecutive NaNs."""
        dates = pd.date_range("2022-01-01", periods=10, freq="B")
        prices = pd.Series([100, np.nan, np.nan, np.nan, np.nan, np.nan, 106, 107, 108, 109],
                           index=dates, name="SPY")
        filled = prices.ffill(limit=5)
        assert filled.iloc[1] == 100  # filled
        assert filled.iloc[5] == 100  # limit reached — still filled (count=5)

    def test_ffill_respects_limit(self):
        """Gaps longer than 5 days should remain NaN after ffill(limit=5)."""
        dates = pd.date_range("2022-01-01", periods=10, freq="B")
        prices = pd.Series([100] + [np.nan] * 8 + [109], index=dates, name="SPY")
        filled = prices.ffill(limit=5)
        assert np.isnan(filled.iloc[6])  # 6th gap exceeds limit
