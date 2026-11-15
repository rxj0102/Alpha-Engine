"""Black-Litterman views and a long-only mean-variance optimiser.

Prior: equilibrium returns PI = delta * Sigma * w_prior. There is no free
source of market capitalisations for this universe, so w_prior is 1/N.

Views are absolute, Q = P mu + noise, with the proportional uncertainty of He
and Litterman (1999): Omega = (1/c - 1) * tau * P Sigma P'. With that choice
tau cancels out of the posterior mean, so it is not a parameter here. If every
asset has a view the posterior is simply PI + c * (Q - PI). Black-Litterman
only does more than shrink when views cover some assets and not others: the
covariance then carries each view over to the correlated assets.
"""
import logging
from typing import Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

log = logging.getLogger(__name__)


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised Ledoit-Wolf shrinkage covariance of daily returns."""
    lw = LedoitWolf().fit(returns.dropna())
    return pd.DataFrame(lw.covariance_ * 252, index=returns.columns, columns=returns.columns)


def equilibrium(cov: pd.DataFrame, risk_aversion: float,
                w_prior: Optional[pd.Series] = None) -> pd.Series:
    """Excess returns under which w_prior is the unconstrained mean-variance optimum."""
    w = pd.Series(1.0 / len(cov), index=cov.index) if w_prior is None else w_prior / w_prior.sum()
    return risk_aversion * cov.dot(w)


def posterior(pi: pd.Series, cov: pd.DataFrame, views: pd.Series, confidence: float) -> pd.Series:
    """Black-Litterman posterior mean for absolute views on a subset of assets.

    views: expected excess return per ticker (the Q vector); tickers without a view are left out.
    """
    views = views.dropna()
    views = views[views.index.isin(pi.index)]
    if views.empty:
        return pi.copy()
    P = np.zeros((len(views), len(pi)))
    for i, t in enumerate(views.index):
        P[i, pi.index.get_loc(t)] = 1.0
    S = cov.values
    k = 1.0 / confidence - 1.0
    # mu = PI + S P' (P S P' + Omega)^-1 (Q - P PI), and Omega = k * P S P' (tau cancels)
    A = (1.0 + k) * (P @ S @ P.T)
    adj = S @ P.T @ np.linalg.solve(A, views.values - P @ pi.values)
    return pd.Series(pi.values + adj, index=pi.index)


def optimize_mv(mu: pd.Series, cov: pd.DataFrame, risk_aversion: float, max_weight: float) -> pd.Series:
    """max mu'w - (delta/2) w'Sigma w  subject to  sum(w) = 1, 0 <= w <= max_weight."""
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(mu.values @ w - (risk_aversion / 2) * cp.quad_form(w, cp.psd_wrap(cov.values))),
        [cp.sum(w) == 1, w >= 0, w <= max_weight],
    )
    for solver in (cp.CLARABEL, cp.OSQP):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None:
                wts = pd.Series(np.clip(w.value, 0, max_weight), index=mu.index)
                return wts / wts.sum()
        except Exception:  # try the next solver
            continue
    log.warning("mean-variance optimisation failed, returning equal weights")
    return pd.Series(1.0 / n, index=mu.index)


def bl_weights(returns: pd.DataFrame, views: Optional[pd.Series], risk_aversion: float,
               max_weight: float, confidence: float) -> Tuple[pd.Series, pd.Series]:
    """Weights and posterior means for one rebalance. views are alphas added to the prior."""
    cov = ledoit_wolf_cov(returns)
    pi = equilibrium(cov, risk_aversion)
    if views is None or views.dropna().empty:
        mu = pi
    else:
        mu = posterior(pi, cov, pi.reindex(views.dropna().index) + views.dropna(), confidence)
    return optimize_mv(mu, cov, risk_aversion, max_weight), mu
