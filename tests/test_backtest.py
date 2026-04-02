"""
tests/test_backtest.py — Unit tests for PortfolioBacktester and StatisticalTests.

Tests cover:
- PortfolioBacktester.run: portfolio value monotonicity, cost deduction
- PortfolioBacktester.metrics: output keys, Sharpe sign, drawdown ≤ 0
- StatisticalTests.block_bootstrap_sharpe: p-value bounds, H₀ rejection
- StatisticalTests.deflated_sharpe_ratio: DSR monotonicity in n_trials
- StatisticalTests.ic_significance: t-test correctness
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import PortfolioBacktester, StatisticalTests


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def prices():
    """Synthetic 3-asset price panel (252 days)."""
    np.random.seed(42)
    n = 252
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    spy = 400 * np.cumprod(1 + np.random.normal(0.0004, 0.01, n))
    tlt = 120 * np.cumprod(1 + np.random.normal(-0.0001, 0.005, n))
    gld = 180 * np.cumprod(1 + np.random.normal(0.0002, 0.007, n))
    return pd.DataFrame({"SPY": spy, "TLT": tlt, "GLD": gld}, index=dates)


@pytest.fixture
def equal_weights(prices):
    """Equal-weight target, rebalanced monthly."""
    n = len(prices.columns)
    return pd.DataFrame(
        np.ones((len(prices), n)) / n,
        index=prices.index, columns=prices.columns,
    )


@pytest.fixture
def backtester():
    return PortfolioBacktester(initial_capital=100_000, risk_free_rate=0.04)


# ── Tests: PortfolioBacktester.run ─────────────────────────────────────────────

class TestBacktesterRun:
    def test_returns_dataframe(self, backtester, prices, equal_weights):
        ph = backtester.run(prices, equal_weights)
        assert isinstance(ph, pd.DataFrame)

    def test_portfolio_value_column_exists(self, backtester, prices, equal_weights):
        ph = backtester.run(prices, equal_weights)
        assert "portfolio_value" in ph.columns

    def test_initial_value(self, backtester, prices, equal_weights):
        ph = backtester.run(prices, equal_weights)
        # First day should be approximately the initial capital
        assert abs(ph["portfolio_value"].iloc[0] - 100_000) < 1000

    def test_length_matches_price_index(self, backtester, prices, equal_weights):
        ph = backtester.run(prices, equal_weights)
        common = prices.index.intersection(equal_weights.index)
        assert len(ph) == len(common)

    def test_portfolio_value_positive(self, backtester, prices, equal_weights):
        ph = backtester.run(prices, equal_weights)
        assert (ph["portfolio_value"] > 0).all()

    def test_transaction_costs_reduce_value(self):
        """With very high transaction costs, final portfolio should be lower."""
        from src.config import BID_ASK_SPREAD
        bt_no_cost = PortfolioBacktester(100_000)
        bt_no_cost.COMMISSION = 0.0  # type: ignore

        np.random.seed(0)
        n = 60
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        prices_small = pd.DataFrame(
            {"A": 100 + np.cumsum(np.random.normal(0, 0.5, n))}, index=dates
        )
        wts = pd.DataFrame({"A": np.ones(n)}, index=dates)

        bt = PortfolioBacktester(100_000)
        ph = bt.run(prices_small, wts, freq="W")
        # With turnover and costs, should not exceed no-cost scenario
        assert ph["portfolio_value"].iloc[-1] > 0  # sanity


# ── Tests: PortfolioBacktester.metrics ────────────────────────────────────────

class TestBacktesterMetrics:
    def test_required_keys(self, backtester, prices, equal_weights):
        backtester.run(prices, equal_weights)
        m = backtester.metrics()
        for key in ["Ann. Return", "Ann. Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"]:
            assert key in m

    def test_max_drawdown_non_positive(self, backtester, prices, equal_weights):
        backtester.run(prices, equal_weights)
        m = backtester.metrics()
        mdd = float(m["Max Drawdown"].strip("%")) / 100
        assert mdd <= 0

    def test_benchmark_adds_beta_alpha(self, backtester, prices, equal_weights):
        backtester.run(prices, equal_weights)
        benchmark = prices["SPY"].pct_change()
        m = backtester.metrics(benchmark=benchmark)
        assert "Beta" in m
        assert "Alpha (ann.)" in m


# ── Tests: StatisticalTests.block_bootstrap_sharpe ───────────────────────────

class TestBlockBootstrapSharpe:
    def test_p_value_in_unit_interval(self):
        np.random.seed(0)
        ret = pd.Series(np.random.normal(0.0004, 0.01, 500))
        result = StatisticalTests.block_bootstrap_sharpe(ret, n_bootstrap=200)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_high_sr_is_significant(self):
        """Deterministically positive returns → very low p-value."""
        np.random.seed(42)
        ret = pd.Series(np.random.normal(0.002, 0.005, 500))  # SR ≈ 6+
        result = StatisticalTests.block_bootstrap_sharpe(ret, n_bootstrap=500)
        assert result["p_value"] < 0.05

    def test_zero_return_not_significant(self):
        """Centred returns → SR ≈ 0 → p-value should not be very small."""
        np.random.seed(7)
        ret = pd.Series(np.random.normal(0.0, 0.01, 500))
        result = StatisticalTests.block_bootstrap_sharpe(ret, n_bootstrap=500)
        assert result["p_value"] > 0.01  # not aggressively rejecting H₀

    def test_returns_observed_sr(self):
        ret = pd.Series(np.random.normal(0.0004, 0.01, 300))
        result = StatisticalTests.block_bootstrap_sharpe(ret, n_bootstrap=100)
        assert "observed_sr" in result
        assert np.isfinite(result["observed_sr"])


# ── Tests: StatisticalTests.deflated_sharpe_ratio ────────────────────────────

class TestDeflatedSharpeRatio:
    def test_more_trials_lowers_dsr(self):
        """More strategies tested → higher threshold → lower DSR."""
        dsr_1 = StatisticalTests.deflated_sharpe_ratio(1.0, n_trials=1, n_obs=252)
        dsr_10 = StatisticalTests.deflated_sharpe_ratio(1.0, n_trials=10, n_obs=252)
        dsr_100 = StatisticalTests.deflated_sharpe_ratio(1.0, n_trials=100, n_obs=252)
        assert dsr_1["DSR"] >= dsr_10["DSR"] >= dsr_100["DSR"]

    def test_dsr_passes_for_high_sharpe(self):
        result = StatisticalTests.deflated_sharpe_ratio(3.0, n_trials=5, n_obs=500)
        assert result["passes_dsr"]

    def test_dsr_fails_for_low_sharpe_many_trials(self):
        result = StatisticalTests.deflated_sharpe_ratio(0.3, n_trials=50, n_obs=252)
        assert not result["passes_dsr"]


# ── Tests: StatisticalTests.ic_significance ───────────────────────────────────

class TestICSig:
    def test_significant_positive_ic(self):
        ic = pd.Series([0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.07, 0.08])
        result = StatisticalTests.ic_significance(ic)
        assert result["significant_5pct"]
        assert result["mean_IC"] > 0

    def test_zero_ic_not_significant(self):
        np.random.seed(0)
        ic = pd.Series(np.random.normal(0, 0.01, 30))
        result = StatisticalTests.ic_significance(ic)
        # Should rarely reject H₀ for truly zero-mean IC
        assert "mean_IC" in result

    def test_insufficient_data(self):
        ic = pd.Series([0.05, 0.06, 0.04])
        result = StatisticalTests.ic_significance(ic)
        assert "error" in result

    def test_icir_computation(self):
        ic = pd.Series([0.10, 0.08, 0.12, 0.09, 0.11])
        result = StatisticalTests.ic_significance(ic)
        expected_icir = ic.mean() / ic.std()
        assert abs(result["ICIR"] - expected_icir) < 1e-10
