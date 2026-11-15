"""End to end run on the frozen data, cut short and with a tiny model so it takes seconds."""
import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.pipeline import FORECAST_STRATEGIES, MARKET, STRATEGIES, run_experiment


@pytest.fixture(scope="module")
def res():
    cfg = Config(end="2000-12-29", gbm_params=dict(n_estimators=3, max_depth=2, random_state=0))
    return run_experiment(cfg, verbose=False)


def test_first_out_of_sample_date(res):
    # 60 training months for the model and 36 months of 20 year yields for carry
    assert res["scores"]["eval_start"] == pd.Timestamp("1996-09-30")


def test_weights_are_valid_for_every_strategy(res):
    for n in STRATEGIES:
        w = res["weights"][n]
        assert w.index[0] == pd.Timestamp("1996-09-30")
        np.testing.assert_allclose(w.sum(axis=1).values, 1.0, atol=1e-8)
        assert (w.values >= -1e-12).all()
        if n in FORECAST_STRATEGIES:
            assert w.values.max() <= res["cfg"].max_weight + 1e-8
    np.testing.assert_allclose(res["weights"]["1/N"].values, 1 / 13)


def test_carry_views_only_change_the_portfolio_through_bonds(res):
    sc = res["scores"]
    assert list(sc["z_carry"].columns) == ["TLT", "IEF", "LQD"]
    w = res["weights"]["Carry (bonds)"]
    assert (w - 1 / 13).abs().max().max() > 0.01


def test_returns_cover_all_strategies_from_the_same_day(res):
    r = res["returns"]
    assert list(r.columns) == STRATEGIES + [MARKET]
    assert r.index[0] == pd.Timestamp("1996-10-01") and r.notna().all().all()
    assert set(res["robust"]) == {"no costs", "double costs", "assumed IC 0.025", "assumed IC 0.10"}
    gross, net = res["robust"]["no costs"]["ML"], r["ML"]
    assert gross.sum() > net.sum()
