"""
optimizer.py — Bayesian portfolio construction via Black-Litterman.

Framework overview
------------------
1. Equilibrium prior  Π = δ·Σ·w_mkt  (reverse-optimisation with market caps)
2. ML views           Q = predicted returns where IC > 0
3. View uncertainty   Ω = (1/c - 1)·τ·P·Σ·Pᵀ  (He & Litterman proportional)
4. BL posterior       μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹·[(τΣ)⁻¹Π + PᵀΩ⁻¹Q]

Key correction [BL-1]
---------------------
τ = 1/T  (T = number of observations, not n_assets).
This scales the prior uncertainty correctly: a 504-day training window
gives τ ≈ 0.002, roughly 14× smaller than the naive τ = 1/n_assets choice,
leading to a less dogmatic posterior and better out-of-sample performance.

Three optimisation modes
-------------------------
- Mean-Variance (CVXPY QP)  : default in low-vol regimes
- Min-CVaR (CVXPY LP)       : used in high-vol regimes (more tail-robust)
- Risk Parity (scipy SLSQP) : equal risk contribution, regime-agnostic fallback
"""

import logging
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.config import OptimizerConfig

log = logging.getLogger(__name__)


class BlackLittermanOptimizer:
    """
    Bayesian portfolio construction with Ledoit-Wolf shrinkage.

    Parameters
    ----------
    cfg : OptimizerConfig
        Risk aversion, risk-free rate, weight bounds, view confidence.
    """

    def __init__(self, cfg: OptimizerConfig = None) -> None:
        self.cfg = cfg or OptimizerConfig()

    # ── Covariance ─────────────────────────────────────────────────────────

    def _cov(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage covariance (annualised).

        Shrinkage is critical for small-T / large-N settings — the sample
        covariance is notoriously ill-conditioned with fewer than ~500 obs
        and 15+ assets.
        """
        lw = LedoitWolf().fit(returns.dropna())
        return pd.DataFrame(
            lw.covariance_ * 252,
            index=returns.columns, columns=returns.columns,
        )

    # ── Black-Litterman ────────────────────────────────────────────────────

    def equilibrium(
        self, returns: pd.DataFrame, market_caps: pd.Series
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Compute the equilibrium return prior via reverse-optimisation.

        Π = δ·Σ·w_mkt

        where δ = risk_aversion and w_mkt is the market-cap-weighted portfolio.
        This encodes the belief that the market portfolio is MV-efficient.

        Returns
        -------
        eq_ret : pd.Series    Equilibrium expected returns.
        cov    : pd.DataFrame Ledoit-Wolf covariance matrix.
        """
        cov = self._cov(returns)
        w_mkt = market_caps.reindex(cov.index).fillna(market_caps.median())
        w_mkt /= w_mkt.sum()
        return self.cfg.risk_aversion * cov.dot(w_mkt), cov

    def posterior(
        self,
        eq_ret: pd.Series,
        cov: pd.DataFrame,
        views: Dict[str, float],
        confidence: Optional[float] = None,
        n_obs: int = 252,
    ) -> pd.Series:
        """
        Compute the Black-Litterman posterior expected returns.

        [BL-1] τ = 1/n_obs  (observation count, not n_assets).

        Parameters
        ----------
        eq_ret     : Equilibrium return vector (Π).
        cov        : Annualised covariance matrix.
        views      : {ticker: predicted_return} — ML views where IC > 0.
        confidence : Analyst confidence ∈ (0, 1); defaults to cfg.view_confidence.
        n_obs      : Training sample size used to scale the prior uncertainty.

        Returns
        -------
        pd.Series
            Posterior expected returns μ_BL.
        """
        c = confidence if confidence is not None else self.cfg.view_confidence
        n_assets = len(eq_ret)
        n_views = len(views)
        tau = 1.0 / max(n_obs, 1)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        for i, (t, v) in enumerate(views.items()):
            if t in eq_ret.index:
                P[i, eq_ret.index.get_loc(t)] = 1.0
                Q[i] = v

        # He & Litterman proportional view uncertainty
        Omega = (1.0 / max(c, 1e-4) - 1.0) * tau * (P @ cov.values @ P.T) + np.eye(n_views) * 1e-8
        tau_sig_inv = np.linalg.inv(tau * cov.values + np.eye(n_assets) * 1e-8)

        try:
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            Omega_inv = np.linalg.pinv(Omega)

        M = tau_sig_inv + P.T @ Omega_inv @ P
        rhs = tau_sig_inv @ eq_ret.values + P.T @ Omega_inv @ Q

        try:
            post = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            post = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return pd.Series(post, index=eq_ret.index)

    # ── Optimisation methods ───────────────────────────────────────────────

    def optimize_mv(
        self,
        mu: pd.Series,
        cov: pd.DataFrame,
        max_weight: Optional[float] = None,
        regime: str = "low_vol",
    ) -> pd.Series:
        """
        Mean-variance quadratic programme (CVXPY).

        Objective: max μᵀw - (λ/2)·wᵀΣw
        Constraints: Σwᵢ = 1, 0 ≤ wᵢ ≤ max_weight

        Regime-aware: λ is doubled in high-vol regimes to discourage
        concentrated bets when uncertainty is elevated.

        Solver cascade: CLARABEL → OSQP → ECOS (robustness).
        """
        lam = self.cfg.risk_aversion * (2.0 if regime == "high_vol" else 1.0)
        mw = max_weight or self.cfg.max_weight
        n = len(mu)
        w = cp.Variable(n)
        prob = cp.Problem(
            cp.Maximize(mu.values @ w - (lam / 2) * cp.quad_form(w, cov.values)),
            [cp.sum(w) == 1, w >= 0, w <= mw],
        )
        for solver in [cp.CLARABEL, cp.OSQP, cp.ECOS]:
            try:
                prob.solve(solver=solver, verbose=False)
                if w.value is not None:
                    wts = pd.Series(np.clip(w.value, 0, mw), index=mu.index)
                    return wts / wts.sum()
            except Exception:
                continue

        log.warning("MV optimisation failed — returning equal weights")
        return pd.Series(1 / n, index=mu.index)

    def optimize_min_cvar(
        self,
        returns: pd.DataFrame,
        alpha: Optional[float] = None,
        max_weight: Optional[float] = None,
    ) -> pd.Series:
        """
        Minimum Conditional Value-at-Risk portfolio (LP formulation).

        CVaR is more robust than MV under fat-tailed, non-normal returns
        and is the preferred objective in high-vol / crisis regimes.

        LP: min η + (1/αT)·Σzₜ
            s.t. zₜ ≥ 0, zₜ ≥ -rₜᵀw - η, Σwᵢ = 1, 0 ≤ wᵢ ≤ max_weight
        """
        a = alpha or self.cfg.cvar_alpha
        mw = max_weight or self.cfg.max_weight
        T, n = returns.shape
        w, eta, z = cp.Variable(n), cp.Variable(), cp.Variable(T)
        prob = cp.Problem(
            cp.Minimize(eta + (1 / (a * T)) * cp.sum(z)),
            [z >= 0, z >= -(returns.values @ w) - eta, cp.sum(w) == 1, w >= 0, w <= mw],
        )
        for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver, verbose=False)
                if w.value is not None:
                    wts = pd.Series(np.clip(w.value, 0, mw), index=returns.columns)
                    return wts / wts.sum()
            except Exception:
                continue

        log.warning("Min-CVaR optimisation failed — returning equal weights")
        return pd.Series(1 / n, index=returns.columns)

    def optimize_risk_parity(self, cov: pd.DataFrame) -> pd.Series:
        """
        Equal Risk Contribution (risk parity) portfolio.

        Each asset contributes σ_portfolio / n to total portfolio risk.
        ERC is parameter-free (no return forecast required) and tends to be
        more stable out-of-sample than mean-variance.

        Solved via SLSQP with a squared-deviation objective.
        """
        n = len(cov)
        Sig = cov.values

        def objective(w):
            vol = np.sqrt(w @ Sig @ w)
            rc = w * (Sig @ w) / vol
            return np.sum((rc - vol / n) ** 2)

        res = minimize(
            objective, np.ones(n) / n, method="SLSQP",
            bounds=[(0.01, self.cfg.max_weight)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if res.success:
            wts = pd.Series(res.x, index=cov.index)
            return wts / wts.sum()
        return pd.Series(1 / n, index=cov.index)
