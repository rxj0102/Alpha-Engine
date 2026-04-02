"""
backtest.py — Event-driven backtester and statistical significance tests.

PortfolioBacktester
--------------------
Simulates a rebalancing strategy with realistic transaction costs.
Cost model: half bid-ask spread (per-ticker) + commission.

[BT-1] Rebalance dates are derived from the actual price index via resample,
       not from a synthetic pd.date_range — this aligns to real trading days
       and handles holiday-adjusted calendars correctly.

StatisticalTests
----------------
Three validation tests to guard against data-mining bias:

1. Block-bootstrap Sharpe  — p-value under H₀: SR = 0; preserves autocorrelation
2. Deflated Sharpe Ratio   — penalises multiple testing (Harvey & Liu 2015)
3. IC t-test               — confirms ML signal has non-zero predictive content
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import BID_ASK_SPREAD, COMMISSION, SEED

log = logging.getLogger(__name__)


class PortfolioBacktester:
    """
    Event-driven backtester with per-ticker transaction cost model.

    Parameters
    ----------
    initial_capital : float
        Starting portfolio value in dollars.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe / Sortino calculation.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.initial_capital = initial_capital
        self.rf_daily = risk_free_rate / 252
        self.portfolio_history = pd.DataFrame()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _tcost(self, ticker: str, notional: float) -> float:
        """Round-trip transaction cost: half spread + commission."""
        spread = BID_ASK_SPREAD.get(ticker, 8e-4)
        return notional * (spread / 2 + COMMISSION)

    @staticmethod
    def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "ME") -> set:
        """
        [BT-1] Derive rebalance dates from the actual price index.

        Resampling the index itself ensures month-end dates land on
        the last real trading day, not a weekend or holiday.
        """
        s = pd.Series(index, index=index)
        try:
            return set(s.resample(freq).last().dropna())
        except Exception:
            return set(s.resample("ME").last().dropna())

    # ── Simulation ─────────────────────────────────────────────────────────

    def run(
        self,
        prices: pd.DataFrame,
        weights_history: pd.DataFrame,
        freq: str = "ME",
    ) -> pd.DataFrame:
        """
        Simulate the portfolio day by day.

        On non-rebalance days holdings drift with price changes.
        On rebalance days, holdings are adjusted to target weights
        after deducting transaction costs from cash.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted daily close prices.
        weights_history : pd.DataFrame
            Date-indexed target weight snapshots (forward-filled externally).
        freq : str
            Resample frequency for rebalance schedule (default 'ME').

        Returns
        -------
        pd.DataFrame
            Daily portfolio value and cash, indexed by date.
        """
        holdings = pd.Series(0.0, index=prices.columns)
        cash = float(self.initial_capital)
        results = []

        common = prices.index.intersection(weights_history.index)
        reb_dates = self._rebalance_dates(common, freq)
        log.info(
            "Backtest: %s → %s | %d rebalances",
            common[0].date(), common[-1].date(), len(reb_dates),
        )

        for date in common:
            px = prices.loc[date]
            pv = float((holdings * px).sum()) + cash

            if date in reb_dates:
                tgt = weights_history.loc[date].reindex(prices.columns).fillna(0)
                tgt /= tgt.sum()
                tgt_holdings = tgt * pv / px.replace(0, np.nan)
                delta = (tgt_holdings - holdings).dropna()
                tcosts = sum(
                    self._tcost(t, abs(d) * px.get(t, 0)) for t, d in delta.items()
                )
                cash = pv - float((tgt_holdings * px).sum()) - tcosts
                holdings = tgt_holdings.fillna(0)

            results.append({"date": date, "portfolio_value": pv, "cash": cash})

        self.portfolio_history = pd.DataFrame(results).set_index("date")
        return self.portfolio_history

    # ── Performance metrics ────────────────────────────────────────────────

    def metrics(self, benchmark: Optional[pd.Series] = None) -> Dict:
        """
        Compute standard risk-adjusted performance statistics.

        Returns
        -------
        dict
            Ann. Return, Ann. Volatility, Sharpe, Sortino, Max Drawdown,
            and optionally Beta / Alpha vs. benchmark.
        """
        ph = self.portfolio_history
        ret = ph["portfolio_value"].pct_change().dropna()
        excess = ret - self.rf_daily
        ann_ret = float(ret.mean() * 252)
        ann_vol = float(ret.std() * np.sqrt(252))
        sharpe = float(excess.mean() * 252 / (excess.std() * np.sqrt(252)))

        cum = (1 + ret).cumprod()
        mdd = float(((cum - cum.expanding().max()) / cum.expanding().max()).min())

        downside = excess[excess < 0]
        sortino = (
            float(excess.mean() * 252 / (downside.std() * np.sqrt(252)))
            if len(downside) > 1 else np.nan
        )

        result = {
            "Ann. Return": f"{ann_ret:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.3f}",
            "Sortino Ratio": f"{sortino:.3f}",
            "Max Drawdown": f"{mdd:.2%}",
        }

        if benchmark is not None:
            b = benchmark.reindex(ret.index).dropna()
            beta = float(np.cov(ret.reindex(b.index), b)[0, 1] / b.var())
            alpha = float(ann_ret - beta * b.mean() * 252)
            result.update({"Beta": f"{beta:.3f}", "Alpha (ann.)": f"{alpha:.2%}"})

        return result


# ── Statistical validation ─────────────────────────────────────────────────────


class StatisticalTests:
    """
    Significance tests to guard against false discoveries in backtesting.

    Methods
    -------
    block_bootstrap_sharpe  — circular block bootstrap p-value for SR > 0
    deflated_sharpe_ratio   — multiple-testing penalty (Harvey & Liu 2015)
    ic_significance         — one-sample t-test on IC time series
    report                  — print a formatted summary of all three tests
    """

    @staticmethod
    def block_bootstrap_sharpe(
        returns: pd.Series,
        n_bootstrap: int = 5000,
        block_size: int = 21,
        rf: float = 0.04,
    ) -> Dict:
        """
        Circular block bootstrap p-value under H₀: Sharpe Ratio = 0.

        Why not a standard t-test?
        --------------------------
        The SR t-test assumes i.i.d. returns. Monthly-rebalanced strategy
        returns are autocorrelated (momentum in holdings, regime persistence),
        so the i.i.d. standard error is downward-biased → the t-test
        over-rejects H₀ (false positives). Block bootstrap preserves the
        autocorrelation structure.

        Circular block bootstrap procedure:
        1. Treat the return series as circular (wrapping around at the end)
        2. Randomly sample starting indices, take blocks of length b = 21
        3. Centre each bootstrap sample at zero (enforces H₀: SR = 0)
        4. Compute bootstrap SR; repeat B = 5000 times
        5. p-value = fraction of bootstrap SRs ≥ observed SR

        Block size b = 21 ≈ 1 trading month. Politis & Romano (1994) suggest
        b ∝ T^(1/3) as the optimal rate; for T=1260 this gives b ≈ 11.
        We use b = 21 to be conservative (preserves more autocorrelation).

        Returns
        -------
        dict
            observed_sr, p_value, significant_5pct
        """
        ex = (returns.dropna() - rf / 252).values
        n = len(ex)
        obs_sr = float(np.sqrt(252) * ex.mean() / ex.std())

        rng = np.random.default_rng(SEED)
        null_srs = []
        for _ in range(n_bootstrap):
            starts = rng.integers(0, n, size=n // block_size + 1)
            idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
            boot = ex[idx]
            boot = boot - boot.mean()  # centre under H₀
            null_srs.append(np.sqrt(252) * boot.mean() / (boot.std() + 1e-10))

        p = float(np.mean(np.array(null_srs) >= obs_sr))
        return {"observed_sr": obs_sr, "p_value": p, "significant_5pct": p < 0.05}

    @staticmethod
    def deflated_sharpe_ratio(
        observed_sharpe: float,
        n_trials: int,
        n_obs: int,
        skewness: float = 0.0,
        excess_kurtosis: float = 0.0,
    ) -> Dict:
        """
        Deflated Sharpe Ratio (Harvey & Liu 2015).

        The problem: if we test N strategy configurations and report the best
        Sharpe, we are guaranteed to find a large SR by chance even if all
        strategies have zero true Sharpe. The DSR adjusts the significance
        threshold upward proportionally to the number of trials.

        The threshold SR under N independent trials is approximated by the
        expected maximum of N standard normals, scaled by 1/√T:

            E[max SR] ≈ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))

        where γ = 0.5772 is the Euler-Mascheroni constant. This is the
        Gumbel approximation to the expected maximum of N i.i.d. normals.

        The threshold is adjusted for non-normality (Harvey & Liu eq. 8):

            SR* = E[max SR]/√T · √(1 - γ₃·SR̂ + (γ₄-1)/4·SR̂²)

        where γ₃ = skewness and γ₄ = excess kurtosis. Negative return skew
        (common in crisis periods) raises the threshold; positive skew lowers it.

        DSR = observed_SR / SR*. DSR > 1 → the Sharpe is significant even
        after penalising for the number of configurations tested.

        Parameters
        ----------
        n_trials : int
            Number of strategy configurations evaluated. In this pipeline,
            n_trials = number of assets with IC > 0 (each asset's ML model
            is a separate "trial"). Conservative lower bound: use 1 to skip
            the penalty entirely; use 50 for a strict test.
        n_obs    : int
            Number of daily return observations in the backtest.
        """
        g = 0.5772  # Euler-Mascheroni constant
        E_max = (
            (1 - g) * stats.norm.ppf(1 - 1 / n_trials)
            + g * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        ) if n_trials > 1 else 0.0

        thr_ann = (
            E_max / np.sqrt(n_obs)
            * np.sqrt(
                1
                - skewness * observed_sharpe
                + (excess_kurtosis - 1) / 4 * observed_sharpe ** 2
            )
            * np.sqrt(252)
        )
        dsr = observed_sharpe / max(thr_ann, 1e-10)
        return {"DSR": dsr, "sr_threshold_ann": thr_ann, "passes_dsr": dsr > 1.0}

    @staticmethod
    def ic_significance(ic_series: pd.Series) -> Dict:
        """
        One-sample t-test on the IC time series (H₀: mean IC = 0).

        ICIR = mean IC / std(IC) — the information ratio of the alpha signal.
        A rule of thumb: ICIR > 0.5 is considered a strong alpha signal.

        Returns
        -------
        dict
            mean_IC, ICIR, t_stat, p_value, significant_5pct
        """
        ic = ic_series.dropna()
        if len(ic) < 5:
            return {"error": "Insufficient observations (need ≥ 5)"}
        t, p = stats.ttest_1samp(ic, popmean=0.0)
        icir = float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan
        return {
            "mean_IC": float(ic.mean()),
            "ICIR": icir,
            "t_stat": float(t),
            "p_value": float(p),
            "significant_5pct": p < 0.05,
        }

    @classmethod
    def report(
        cls,
        portfolio_history: pd.DataFrame,
        all_ml_metrics: Dict,
        n_trials: int = 10,
        rf: float = 0.04,
    ) -> None:
        """Print a formatted statistical significance report to stdout."""
        ret = portfolio_history["portfolio_value"].pct_change().dropna()

        print("\n" + "=" * 55)
        print("  STATISTICAL SIGNIFICANCE REPORT")
        print("=" * 55)

        bbs = cls.block_bootstrap_sharpe(ret, rf=rf)
        print(f"\n[1] Block-Bootstrap Sharpe (H₀: SR=0, B=5000, block=21d)")
        print(
            f"    SR={bbs['observed_sr']:.3f}  p={bbs['p_value']:.4f}  "
            f"{'✓ Significant' if bbs['significant_5pct'] else '✗ Not significant'} at 5%"
        )

        dsr = cls.deflated_sharpe_ratio(
            bbs["observed_sr"], n_trials, len(ret),
            skewness=float(ret.skew()), excess_kurtosis=float(ret.kurt()),
        )
        print(f"\n[2] Deflated Sharpe Ratio (Harvey & Liu 2015, n_trials={n_trials})")
        print(
            f"    DSR={dsr['DSR']:.3f}  threshold={dsr['sr_threshold_ann']:.3f}  "
            f"{'✓ Passes' if dsr['passes_dsr'] else '✗ Fails'}"
        )

        if all_ml_metrics:
            ic_vals = pd.Series(
                {t: m.get("IC", np.nan) for t, m in all_ml_metrics.items()}
            ).dropna()
            ic_res = cls.ic_significance(ic_vals)
            if "error" not in ic_res:
                print(f"\n[3] ML Signal IC (H₀: IC=0, n={len(ic_vals)} assets)")
                print(
                    f"    mean_IC={ic_res['mean_IC']:.4f}  ICIR={ic_res['ICIR']:.3f}  "
                    f"p={ic_res['p_value']:.4f}  "
                    f"{'✓ Significant' if ic_res['significant_5pct'] else '✗ Not significant'} at 5%"
                )

        print("=" * 55)
