import numpy as np
import pandas as pd

from src.config import Config
from src.data import FEATURES, all_features, asset_features, load_panel, month_ends


def test_features_are_causal(prices):
    """A feature on day t must not change when later prices are removed."""
    full = asset_features(prices["C"])
    cut = 900
    part = asset_features(prices["C"].iloc[: cut + 1])
    pd.testing.assert_series_equal(full.iloc[cut], part.iloc[cut])
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], part)


def test_feature_set_and_warm_up(prices):
    f = asset_features(prices["A"])
    assert list(f.columns) == FEATURES and len(FEATURES) == 18
    assert f.iloc[:252].isna().all().all()
    assert f.iloc[300:].notna().all().all()


def test_features_do_not_depend_on_price_level(prices):
    a, b = asset_features(prices["B"]), asset_features(prices["B"] * 37.0)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_late_starting_asset_gets_its_own_warm_up(prices):
    p = prices["D"].copy()
    p.iloc[:400] = np.nan
    f = asset_features(p)
    assert f.iloc[:652].isna().all().all() and f.iloc[660:].notna().all().all()


def test_month_ends_are_last_trading_days(prices):
    me = month_ends(prices.index)
    assert me.isin(prices.index).all()
    nxt = prices.index[prices.index.get_indexer(me[:-1]) + 1]
    assert (nxt.month != me[:-1].month).all()


def test_snapshot_panel():
    cfg = Config()
    panel = load_panel(cfg)
    px = panel["prices"]
    assert list(px.columns) == cfg.universe
    assert px.index[0] == pd.Timestamp("1990-07-02") and px.index.is_monotonic_increasing
    assert px.drop(columns="TLT").notna().all().all()
    assert px["TLT"].first_valid_index() == pd.Timestamp("1993-10-01")
    assert panel["rf"].notna().all() and (panel["rf"] >= 0).all()
    assert set(panel["yields"].columns) == {"DGS7", "DGS10", "DGS20", "DBAA"}
    assert len(all_features(px.iloc[:400])) == 13
