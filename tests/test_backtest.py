import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import backtest as bt
from src.data import month_ends


@pytest.fixture(scope="module")
def targets(prices):
    reb = month_ends(prices.index)[1:-1]
    return pd.DataFrame(1 / 6, index=reb, columns=prices.columns)


def test_returns_start_the_day_after_the_first_rebalance(prices, targets):
    r = bt.backtest(prices, targets)["returns"]
    assert r.index[0] == prices.index[prices.index.get_loc(targets.index[0]) + 1]
    assert r.index[-1] == prices.index[-1]


def test_single_asset_matches_its_own_return(prices, targets):
    t = targets * 0.0
    t["B"] = 1.0
    r = bt.backtest(prices, t, cost_mult=1.0)
    np.testing.assert_allclose(r["returns"].values, prices["B"].pct_change().loc[r["returns"].index].values, atol=1e-12)
    assert (r["turnover"] < 1e-12).all()


def test_first_period_is_the_weighted_asset_return(prices, targets):
    r = bt.backtest(prices, targets, cost_mult=0.0)["returns"]
    a, b = targets.index[0], targets.index[1]
    want = (prices.loc[b] / prices.loc[a] - 1).mean()
    np.testing.assert_allclose((1 + r.loc[:b]).prod() - 1, want, rtol=1e-10)


def test_costs_equal_turnover_times_cost(prices, targets):
    rng = np.random.default_rng(3)
    w = pd.DataFrame(rng.dirichlet(np.ones(6), len(targets)), index=targets.index, columns=targets.columns)
    free, paid = bt.backtest(prices, w, 0.0), bt.backtest(prices, w, 1.0)
    assert (paid["returns"] <= free["returns"] + 1e-15).all()
    d = w.index[3]
    cost = bt.cost_vector(prices.columns)
    assert np.allclose(cost, 8e-4 + 5e-4)           # synthetic tickers use the default spread
    drag = 1 - (1 + paid["returns"].loc[d]) / (1 + free["returns"].loc[d])
    np.testing.assert_allclose(drag, 2 * paid["turnover"].loc[d] * cost[0], rtol=1e-6)
    assert 0 < paid["turnover"].mean() < 1


def test_summarize_uses_the_bill_rate(prices, targets):
    r = bt.backtest(prices, targets)["returns"]
    zero, some = pd.Series(0.0, index=r.index), pd.Series(0.04 / 252, index=r.index)
    a, b = bt.summarize(r, zero), bt.summarize(r, some)
    np.testing.assert_allclose(a["ann_excess"] - b["ann_excess"], 0.04, rtol=1e-9)
    assert a["sharpe"] > b["sharpe"] and a["max_drawdown"] < 0 and a["n_days"] == len(r)


def test_sharpe_interval_contains_the_estimate():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.0004, 0.01, 2500))
    ci = bt.sharpe_ci(x, n_boot=500)
    assert ci["lo"] < ci["sharpe"] < ci["hi"]
    assert 0.5 < ci["hi"] - ci["lo"] < 2.5          # about 4 / sqrt(10 years)


def test_sharpe_difference():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0003, 0.01, 3000)
    a = pd.Series(base)
    same = bt.sharpe_diff_test(a, a, n_boot=200)
    assert same["diff"] == 0 and same["p_value"] == 1.0
    better = pd.Series(base + 0.002)
    out = bt.sharpe_diff_test(better, a, n_boot=500)
    assert out["diff"] > 2 and out["lo"] > 0 and out["p_value"] < 0.01
    noise = pd.Series(base + rng.normal(0, 0.002, 3000))
    assert bt.sharpe_diff_test(noise, a, n_boot=500)["p_value"] > 0.05


def test_deflated_sharpe_ratio():
    # one trial, normal returns: the probabilistic Sharpe ratio against zero
    out = bt.deflated_sharpe(0.1, 121, 0.0, 3.0)
    np.testing.assert_allclose(out["dsr"], stats.norm.cdf(0.1 * np.sqrt(120) / np.sqrt(1 + 0.5 * 0.01)))
    assert bt.deflated_sharpe(0.0, 100, 0.0, 3.0)["dsr"] == 0.5
    # more trials or more dispersion across trials raise the hurdle
    a = bt.deflated_sharpe(0.1, 121, 0.0, 3.0, n_trials=5, trials_sr_std=0.05)
    b = bt.deflated_sharpe(0.1, 121, 0.0, 3.0, n_trials=50, trials_sr_std=0.05)
    assert out["dsr"] > a["dsr"] > b["dsr"] and 0 < a["sr0"] < b["sr0"]
    # expected maximum of 5 standard normals is about 1.16; the approximation gives 1.19
    assert abs(a["sr0"] / 0.05 - 1.163) < 0.05
    # negative skew and fat tails lower the confidence in a positive Sharpe ratio
    assert bt.deflated_sharpe(0.1, 121, -1.0, 8.0)["dsr"] < out["dsr"]


def test_rank_ic():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-31", "2020-02-28"]), list("abcdef")],
                                     names=["date", "ticker"])
    s = pd.Series(np.tile(np.arange(6.0), 2), index=idx)
    o = pd.Series(np.r_[np.arange(6.0), -np.arange(6.0)], index=idx)
    ic = bt.rank_ic(s, o)
    assert ic.iloc[0] == 1.0 and ic.iloc[1] == -1.0
    assert bt.rank_ic(s[:4], o[:4]).empty           # fewer than 5 assets


def test_newey_west():
    rng = np.random.default_rng(2)
    x = rng.normal(0.1, 1, 400)
    np.testing.assert_allclose(bt.nw_tstat(x, 0), stats.ttest_1samp(x, 0).statistic, rtol=2e-3)
    ar = np.zeros(2000)
    e = rng.normal(0, 1, 2000)
    for i in range(1, 2000):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    ar += 0.3
    assert abs(bt.nw_tstat(ar, 12)) < abs(bt.nw_tstat(ar, 0))   # positive autocorrelation widens the error
    s = bt.ic_stats(pd.Series(x))
    np.testing.assert_allclose(s["detectable_ic"], 2 * x.std(ddof=1) / 20)
    assert s["n"] == 400 and 0 < s["p_value"] < 1


def test_monthly_compounding():
    idx = pd.bdate_range("2020-01-01", "2020-02-28")
    r = pd.Series(0.01, index=idx)
    m = bt.monthly(r)
    np.testing.assert_allclose(m.iloc[0], 1.01 ** (idx.month == 1).sum() - 1)
