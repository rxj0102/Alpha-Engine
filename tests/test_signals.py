import numpy as np
import pandas as pd

from src import signals
from src.opendata import par_bond_duration


def test_momentum_skips_the_latest_month(prices):
    bumped = prices.copy()
    bumped.iloc[-1] *= 1.5          # a jump on the last day only
    a, b = signals.momentum(prices).iloc[-2], signals.momentum(bumped).iloc[-2]
    pd.testing.assert_series_equal(a, b)
    t = prices.index[-1]
    ret = prices.iloc[-22] / prices.iloc[-253] - 1
    vol = prices.pct_change().iloc[-252:].std() * np.sqrt(252)
    np.testing.assert_allclose(signals.momentum(prices).loc[t], ret / vol)


def test_xs_zscore():
    z = signals.xs_zscore(pd.Series([1.0, 2.0, 3.0, np.nan, 100.0], index=list("abcde")), clip=1.2)
    assert np.isnan(z["d"]) and z.max() <= 1.2
    assert signals.xs_zscore(pd.Series([1.0, 1.0, 1.0])).isna().all()
    np.testing.assert_allclose(signals.xs_zscore(pd.Series([1.0, 2.0, 3.0, 4.0])).mean(), 0, atol=1e-12)


def test_ts_zscore_uses_history_only():
    h = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 10.0], "y": [np.nan, np.nan, np.nan, 1.0, 2.0]})
    z = signals.ts_zscore(h, min_obs=3)
    np.testing.assert_allclose(z["x"], (10 - h["x"].mean()) / h["x"].std())
    assert np.isnan(z["y"])


def test_alpha_scaling():
    a = signals.alpha_from_z(pd.Series({"a": 1.0, "b": -2.0}), pd.Series({"a": 0.2, "b": 0.1}), ic=0.05)
    np.testing.assert_allclose(a["a"], 0.05 * 0.2 * np.sqrt(12))
    np.testing.assert_allclose(a["b"], -2 * 0.05 * 0.1 * np.sqrt(12))


def test_carry_on_a_flat_curve_is_yield_minus_bill():
    idx = pd.bdate_range("2020-01-01", periods=3)
    y = pd.DataFrame({"DGS7": 4.0, "DGS10": 4.0, "DGS20": 4.0, "DBAA": 6.0}, index=idx)
    c = signals.bond_carry(y, pd.Series(0.02 / 252, index=idx))
    np.testing.assert_allclose(c["IEF"], 0.02)
    np.testing.assert_allclose(c["TLT"], 0.02)
    np.testing.assert_allclose(c["LQD"], 0.04)


def test_carry_adds_roll_down_on_a_rising_curve():
    idx = pd.bdate_range("2020-01-01", periods=2)
    y = pd.DataFrame({"DGS7": 3.7, "DGS10": 4.0, "DGS20": 4.5, "DBAA": 6.0}, index=idx)
    c = signals.bond_carry(y, pd.Series(0.0, index=idx))
    d10 = float(par_bond_duration(pd.Series([0.04]), 10).iloc[0])
    np.testing.assert_allclose(c["IEF"].iloc[0], 0.04 + d10 * 0.001)
    assert 7.5 < d10 < 8.5


def test_combine_averages_what_exists():
    out = signals.combine([pd.Series({"a": 1.0, "b": 2.0}), pd.Series({"a": 3.0})])
    assert out["a"] == 2.0 and out["b"] == 2.0
