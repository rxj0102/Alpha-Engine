import numpy as np
import pandas as pd
import pytest

from src import ml_alpha, signals
from src.config import Config
from src.data import FEATURES, all_features, month_ends


@pytest.fixture(scope="module")
def panel(prices):
    vol = signals.realised_vol(prices, 63)
    reb = month_ends(prices.index)
    reb = reb[prices.index.get_indexer(reb) >= 260]
    X, y, fwd = ml_alpha.build_panel(all_features(prices), prices, vol, reb)
    return X, y, fwd, reb, vol


def small_cfg(**kw):
    return Config(min_train_months=12, gbm_params=dict(n_estimators=5, max_depth=2, random_state=0), **kw)


def test_panel_shape_and_ranks(panel, prices):
    X, y, fwd, reb, _ = panel
    assert list(X.columns) == FEATURES
    assert len(X) == len(reb) * prices.shape[1]
    assert X.abs().max().max() < 0.5
    np.testing.assert_allclose(X.groupby(level="date").mean().values, 0, atol=1e-12)


def test_target_is_relative_and_missing_on_last_date(panel):
    X, y, fwd, reb, _ = panel
    np.testing.assert_allclose(y.dropna().groupby(level="date").mean().values, 0, atol=1e-12)
    assert y.xs(reb[-1], level="date").isna().all()
    assert y.xs(reb[-2], level="date").notna().all()


def test_target_is_next_month_return_over_lagged_vol(panel, prices):
    X, y, fwd, reb, vol = panel
    d, nxt = reb[5], reb[6]
    prev = prices.index[prices.index.get_loc(d) - 1]
    want = (prices.loc[nxt, "C"] / prices.loc[d, "C"] - 1) / vol.loc[prev, "C"]
    np.testing.assert_allclose(fwd.loc[(d, "C")], want)


def test_features_exclude_the_rebalance_day(prices):
    """Changing the price on a rebalance day must not change that day's feature row."""
    vol = signals.realised_vol(prices, 63)
    reb = month_ends(prices.index)
    reb = reb[prices.index.get_indexer(reb) >= 260]
    bumped = prices.copy()
    bumped.loc[reb[10]] *= 1.2
    X0, _, _ = ml_alpha.build_panel(all_features(prices), prices, vol, reb)
    X1, _, _ = ml_alpha.build_panel(all_features(bumped), bumped, signals.realised_vol(bumped, 63), reb)
    pd.testing.assert_frame_equal(X0.xs(reb[10], level="date"), X1.xs(reb[10], level="date"))


def test_walk_forward_starts_after_min_history(panel):
    X, y, _, reb, _ = panel
    fc = ml_alpha.walk_forward(X, y, small_cfg())
    first = fc.index.get_level_values("date").min()
    assert first == reb[13]          # 12 training months plus the one purged month
    assert list(fc.columns) == ml_alpha.MODELS


def test_forecast_ignores_the_purged_and_future_labels(panel):
    """Labels from the latest month onward must not affect the forecast for a date."""
    X, y, _, reb, _ = panel
    cfg = small_cfg()
    base = ml_alpha.walk_forward(X, y, cfg)
    k = 20
    y2 = y.copy()
    later = y2.index.get_level_values("date") >= reb[k - 1]
    y2[later] = y2[later] + 50.0 * np.sign(np.arange(later.sum()) % 2 - 0.5)
    pert = ml_alpha.walk_forward(X, y2, cfg)
    pd.testing.assert_frame_equal(base.xs(reb[k], level="date"), pert.xs(reb[k], level="date"))
    assert not np.allclose(base.xs(reb[k + 3], level="date")["ridge"], pert.xs(reb[k + 3], level="date")["ridge"])


def test_recovers_a_planted_signal(panel):
    X, y, _, _, _ = panel
    rng = np.random.default_rng(1)
    planted = (2.0 * X["mom_63d"] + rng.normal(0, 0.2, len(X))).where(y.notna())
    fc = ml_alpha.walk_forward(X, planted, small_cfg())
    df = pd.concat([fc["ridge"], planted], axis=1).dropna()
    assert df.corr(method="spearman").iloc[0, 1] > 0.5


def test_forecasts_are_reproducible(panel):
    X, y, _, _, _ = panel
    cfg = small_cfg()
    pd.testing.assert_frame_equal(ml_alpha.walk_forward(X, y, cfg), ml_alpha.walk_forward(X, y, cfg))
