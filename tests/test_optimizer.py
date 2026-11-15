import numpy as np
import pandas as pd
import pytest

from src import optimizer as opt


@pytest.fixture(scope="module")
def cov(prices):
    return opt.ledoit_wolf_cov(prices.pct_change().dropna())


def test_cov_is_annualised_and_positive_definite(cov, prices):
    assert np.all(np.linalg.eigvalsh(cov.values) > 0)
    sample = prices.pct_change().var() * 252
    np.testing.assert_allclose(np.diag(cov.values), sample.values, rtol=0.25)


def test_no_view_portfolio_is_the_prior(cov):
    """Reverse optimisation: with equilibrium returns the optimiser gives back 1/N."""
    pi = opt.equilibrium(cov, 2.5)
    w = opt.optimize_mv(pi, cov, 2.5, 0.30)
    np.testing.assert_allclose(w.values, 1 / len(cov), atol=1e-5)


def test_posterior_without_views_is_the_prior(cov):
    pi = opt.equilibrium(cov, 2.5)
    pd.testing.assert_series_equal(opt.posterior(pi, cov, pd.Series(dtype=float), 0.65), pi)


def test_full_views_shrink_towards_the_prior_by_confidence(cov):
    pi = opt.equilibrium(cov, 2.5)
    q = pi + pd.Series(np.linspace(-0.03, 0.03, len(pi)), index=pi.index)
    for c in (0.25, 0.65, 0.9):
        np.testing.assert_allclose(opt.posterior(pi, cov, q, c).values, (pi + c * (q - pi)).values, atol=1e-12)


def test_posterior_matches_the_textbook_formula_for_any_tau(cov):
    """The general Black-Litterman formula gives the same mean whatever tau is."""
    pi = opt.equilibrium(cov, 2.5)
    views = pd.Series({"B": pi["B"] + 0.02, "E": pi["E"] - 0.03})
    c, S = 0.65, cov.values
    P = np.zeros((2, len(pi)))
    P[0, 1] = P[1, 4] = 1.0
    for tau in (0.002, 0.05, 1.0):
        om = (1 / c - 1) * tau * P @ S @ P.T
        A = np.linalg.inv(tau * S)
        mu = np.linalg.solve(A + P.T @ np.linalg.inv(om) @ P, A @ pi.values + P.T @ np.linalg.inv(om) @ views.values)
        np.testing.assert_allclose(opt.posterior(pi, cov, views, c).values, mu, rtol=1e-8)


def test_partial_view_moves_correlated_assets(cov):
    pi = opt.equilibrium(cov, 2.5)
    mu = opt.posterior(pi, cov, pd.Series({"F": pi["F"] + 0.05}), 0.65)
    np.testing.assert_allclose(mu["F"] - pi["F"], 0.65 * 0.05)
    assert (mu.drop("F") > pi.drop("F")).all()     # all assets share a positive common factor


def test_weights_respect_the_constraints(cov):
    pi = opt.equilibrium(cov, 2.5)
    mu = pi.copy()
    mu["A"] += 0.5
    w = opt.optimize_mv(mu, cov, 2.5, 0.30)
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= 0 and w.max() <= 0.30 + 1e-9
    assert abs(w["A"] - 0.30) < 1e-6


def test_bl_weights_tilt_towards_positive_views(prices):
    r = prices.pct_change().dropna()
    w0, _ = opt.bl_weights(r, None, 2.5, 0.30, 0.65)
    w1, _ = opt.bl_weights(r, pd.Series({"C": 0.02, "D": -0.02}), 2.5, 0.30, 0.65)
    np.testing.assert_allclose(w0.values, 1 / 6, atol=1e-5)
    assert w1["C"] > w0["C"] and w1["D"] < w0["D"]
