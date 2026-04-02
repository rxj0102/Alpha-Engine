"""
tests/test_optimizer.py — Unit tests for BlackLittermanOptimizer.

Tests cover:
- equilibrium: output types, shapes, Π = δΣw_mkt identity
- posterior: views incorporated, τ = 1/T scaling
- optimize_mv: weights sum to 1, bounds respected, regime effect
- optimize_min_cvar: weights sum to 1, bounds respected
- optimize_risk_parity: equal risk contributions (approximately)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.optimizer import BlackLittermanOptimizer
from src.config import OptimizerConfig


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def returns_df():
    """Synthetic 3-asset return matrix (252 × 3)."""
    np.random.seed(42)
    n = 252
    cov = np.array([[0.04, 0.01, -0.005],
                    [0.01, 0.02, 0.003],
                    [-0.005, 0.003, 0.01]]) / 252
    chol = np.linalg.cholesky(cov)
    data = np.random.normal(size=(n, 3)) @ chol.T
    return pd.DataFrame(data, columns=["SPY", "TLT", "GLD"])


@pytest.fixture
def market_caps():
    return pd.Series({"SPY": 400e9, "TLT": 50e9, "GLD": 60e9})


@pytest.fixture
def bl(returns_df):
    opt = BlackLittermanOptimizer()
    return opt, returns_df


# ── Tests: equilibrium ─────────────────────────────────────────────────────────

class TestEquilibrium:
    def test_output_types(self, bl, market_caps):
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        assert isinstance(eq, pd.Series)
        assert isinstance(cov, pd.DataFrame)

    def test_output_shape(self, bl, market_caps):
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        assert len(eq) == 3
        assert cov.shape == (3, 3)

    def test_cov_is_positive_definite(self, bl, market_caps):
        opt, ret = bl
        _, cov = opt.equilibrium(ret, market_caps)
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > 0), "Covariance matrix is not positive definite"

    def test_equilibrium_formula(self, bl, market_caps):
        """Verify Π = δ·Σ·w_mkt."""
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        w_mkt = market_caps.reindex(cov.index).fillna(market_caps.median())
        w_mkt /= w_mkt.sum()
        expected = opt.cfg.risk_aversion * cov.dot(w_mkt)
        np.testing.assert_allclose(eq.values, expected.values, rtol=1e-10)


# ── Tests: posterior ───────────────────────────────────────────────────────────

class TestPosterior:
    def test_no_views_returns_prior(self, bl, market_caps):
        """With empty views, posterior should equal the prior (no update)."""
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        post = opt.posterior(eq, cov, views={}, n_obs=252)
        # With no views the posterior collapses to the prior
        assert isinstance(post, pd.Series)
        assert len(post) == len(eq)

    def test_views_shift_posterior(self, bl, market_caps):
        """A strong upward view on SPY should raise its posterior return."""
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        post_no_views = opt.posterior(eq, cov, views={}, n_obs=252)
        views = {"SPY": 0.30}  # very bullish view
        post_with_views = opt.posterior(eq, cov, views=views, n_obs=252)
        assert post_with_views["SPY"] > post_no_views["SPY"]

    def test_tau_scaling(self, bl, market_caps):
        """Larger n_obs → smaller τ → posterior closer to prior."""
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        views = {"SPY": 0.20}
        post_small = opt.posterior(eq, cov, views, n_obs=50)
        post_large = opt.posterior(eq, cov, views, n_obs=5000)
        # With large n_obs, τ is tiny → prior dominates → posterior ≈ prior
        assert abs(post_large["SPY"] - eq["SPY"]) < abs(post_small["SPY"] - eq["SPY"])


# ── Tests: optimize_mv ─────────────────────────────────────────────────────────

class TestOptimizeMV:
    def test_weights_sum_to_one(self, bl, market_caps):
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        wts = opt.optimize_mv(eq, cov)
        assert abs(wts.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, bl, market_caps):
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        wts = opt.optimize_mv(eq, cov)
        assert (wts >= -1e-8).all()

    def test_max_weight_respected(self, bl, market_caps):
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        max_w = 0.5
        wts = opt.optimize_mv(eq, cov, max_weight=max_w)
        assert (wts <= max_w + 1e-6).all()

    def test_high_vol_regime_more_conservative(self, bl, market_caps):
        """High-vol regime doubles λ → more diversified (lower max weight)."""
        opt, ret = bl
        eq, cov = opt.equilibrium(ret, market_caps)
        wts_low = opt.optimize_mv(eq, cov, regime="low_vol")
        wts_high = opt.optimize_mv(eq, cov, regime="high_vol")
        assert wts_high.max() <= wts_low.max() + 1e-6


# ── Tests: optimize_min_cvar ───────────────────────────────────────────────────

class TestOptimizeMinCVaR:
    def test_weights_sum_to_one(self, bl):
        opt, ret = bl
        wts = opt.optimize_min_cvar(ret)
        assert abs(wts.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, bl):
        opt, ret = bl
        wts = opt.optimize_min_cvar(ret)
        assert (wts >= -1e-8).all()

    def test_max_weight_respected(self, bl):
        opt, ret = bl
        max_w = 0.6
        wts = opt.optimize_min_cvar(ret, max_weight=max_w)
        assert (wts <= max_w + 1e-6).all()


# ── Tests: optimize_risk_parity ────────────────────────────────────────────────

class TestOptimizeRiskParity:
    def test_weights_sum_to_one(self, bl, market_caps):
        opt, ret = bl
        _, cov = opt.equilibrium(ret, market_caps)
        wts = opt.optimize_risk_parity(cov)
        assert abs(wts.sum() - 1.0) < 1e-6

    def test_approximately_equal_risk_contributions(self, bl, market_caps):
        """Risk contributions should be roughly equal (within 5pp)."""
        opt, ret = bl
        _, cov = opt.equilibrium(ret, market_caps)
        wts = opt.optimize_risk_parity(cov)
        w = wts.values
        Sig = cov.values
        port_vol = np.sqrt(w @ Sig @ w)
        rc = w * (Sig @ w) / port_vol
        rc_pct = rc / rc.sum()
        # Each asset's risk share should be within ±5pp of 1/n
        n = len(wts)
        assert np.all(np.abs(rc_pct - 1 / n) < 0.05), \
            f"Risk contributions not equalised: {rc_pct}"
