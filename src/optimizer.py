"""
optimizer.py — Bayesian portfolio construction via Black-Litterman.

Black-Litterman in one equation
--------------------------------
The BL posterior is the precision-weighted average of prior and views:

    μ_BL = M⁻¹ · rhs
    M    = (τΣ)⁻¹ + PᵀΩ⁻¹P          (posterior precision)
    rhs  = (τΣ)⁻¹Π + PᵀΩ⁻¹Q         (precision-weighted mean)

This is just Bayes' theorem with Gaussian distributions. The prior is
N(Π, τΣ); the view likelihood is N(Q, Ω); the posterior mean is the
standard formula for the product of two Gaussians.

Components
----------
1. Equilibrium prior  Π = δ·Σ·w_mkt
   Reverse-optimisation: if the market portfolio is MV-efficient, the
   implied expected returns are Π = δΣw_mkt. This encodes the CAPM
   prior without requiring any alpha forecast.

2. Prior uncertainty  τ = 1/T  [BL-1]
   τΣ is the uncertainty in the prior mean. The choice τ = 1/T comes
   from treating Π as a sample mean estimated from T observations; the
   standard error of a sample mean is Σ/T. The common heuristic τ = 1/N
   has no statistical basis and overstates prior certainty by T/N ≈ 34×
   for our 504-day window with 15 assets.

3. View matrix P and view vector Q
   Each row of P selects the assets covered by one view. We use absolute
   views (P is a row of the identity), so each ML forecast directly
   specifies the expected return of one asset, not a relative spread.

4. View uncertainty  Ω = (1/c - 1)·τ·PΣPᵀ  (He & Litterman proportional)
   Ω scales the view uncertainty relative to the prior uncertainty in the
   same view direction (PΣPᵀ). At c = 0.5, Ω = τPΣPᵀ (equal weight to
   prior and views). At c → 1, Ω → 0 and views completely override the
   prior. We set c = 0.65 (modest trust in ML forecasts).

Three optimisation modes
-------------------------
- Mean-Variance (CVXPY QP)  : default in low-vol regimes
- Min-CVaR (CVXPY LP)       : used in high-vol regimes (more tail-robust)
- Risk Parity (scipy SLSQP) : equal risk contribution, regime-agnostic fallback

Numerical notes
---------------
- Ledoit-Wolf guarantees Σ is positive definite, so (τΣ)⁻¹ is well-defined.
- We add ε·I (ε = 1e-8) to Ω to guard against numerical singularity when
  all views are on uncorrelated assets (Ω diagonal and near-zero).
- np.linalg.solve is preferred over np.linalg.inv for the posterior solve
  (O(n²) vs O(n³) for symmetric systems; falls back to lstsq if singular).
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

        The sample covariance S = (1/T)XᵀX has estimation error of order
        N/T. For N=15 assets and T=504 days, N/T ≈ 0.03 — manageable but
        non-trivial, and MV optimisation amplifies small covariance errors
        into large weight errors (the "error maximisation" problem, Michaud 1989).

        Ledoit-Wolf (2004) shrinks toward a structured target (here: constant
        correlation) using the analytically optimal shrinkage intensity α*:

            Σ_LW = (1 - α*)S + α*μ_S · I

        where μ_S = tr(S)/N is the average variance. The Oracle estimator
        minimises E[‖Σ_LW - Σ_true‖²_F], trading bias for variance reduction.
        For our T/N ≈ 34 ratio, α* ≈ 0.05–0.15 (mild shrinkage).
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
        try:
            tau_sig_inv = np.linalg.inv(tau * cov.values + np.eye(n_assets) * 1e-8)
        except np.linalg.LinAlgError:
            tau_sig_inv = np.linalg.pinv(tau * cov.values + np.eye(n_assets) * 1e-8)

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

        The unconstrained MV solution is w* = (1/λ)Σ⁻¹μ, which is the
        tangency portfolio scaled by risk aversion. With the long-only and
        weight-cap constraints the QP has no closed form but remains convex:

            max  μᵀw - (λ/2)·wᵀΣw
            s.t. 1ᵀw = 1,  0 ≤ wᵢ ≤ w̄

        Regime-aware λ: in high-vol regimes we double λ (= 5.0 vs 2.5).
        This is equivalent to a 50% reduction in the perceived information
        ratio of the forecast, reflecting the empirical finding that ML
        signals have lower Sharpe during high-dispersion markets.

        Solver cascade: CLARABEL (interior-point, default in CVXPY ≥ 1.4)
        → OSQP (ADMM, robust to ill-conditioning) → ECOS (barrier method,
        legacy fallback). All three produce identical solutions to within 1e-6.
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

        CVaR (Expected Shortfall) at level α is the expected loss conditional
        on the loss exceeding the α-quantile (VaR). Unlike VaR, CVaR is a
        coherent risk measure (Artzner et al. 1999): it is sub-additive, so
        diversification always reduces it. MV is not coherent — it penalises
        upside variance equally with downside variance.

        The Rockafellar-Uryasev (2000) LP reformulation avoids the
        non-convexity of directly minimising the empirical quantile:

            CVaR_α(w) = min_{η} { η + (1/αT) Σ_t max(-rₜᵀw - η, 0) }

        Introducing z_t = max(-rₜᵀw - η, 0) as a variable:

            min  η + (1/αT)·Σzₜ
            s.t. zₜ ≥ 0,  zₜ ≥ -rₜᵀw - η,  1ᵀw = 1,  0 ≤ wᵢ ≤ w̄

        At α = 0.05 this minimises the average loss in the worst 5% of
        days — appropriate when the regime model signals elevated tail risk.
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

        The risk contribution of asset i is:

            RC_i = w_i · (∂σ_p/∂w_i) = w_i · (Σw)_i / σ_p

        ERC requires RC_i = σ_p / n for all i, i.e. the gradient of portfolio
        vol with respect to each weight is equal across all assets. This has
        no closed-form solution for N > 2 but is convex in the objective:

            min_w  Σ_i Σ_j (RC_i - RC_j)² = Σ_i (RC_i - σ_p/n)²

        Equivalent to a Sharpe-maximising portfolio when all assets have equal
        Sharpe ratios. Practically, ERC is parameter-free (no μ required),
        weights are more stable than MV (lower turnover), and it avoids the
        error amplification problem that plagues unconstrained MV.

        SLSQP bounds (0.01, max_weight) prevent degenerate solutions when the
        covariance matrix has very low off-diagonal entries.
        """
        n = len(cov)
        Sig = cov.values

        def objective(w):
            vol = np.sqrt(w @ Sig @ w)
            if vol < 1e-10:
                return 1e10  # degenerate portfolio; penalise heavily
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
