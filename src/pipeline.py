"""
pipeline.py — Walk-forward orchestration of the full Alpha Engine pipeline.

Pipeline stages per rebalance date
------------------------------------
A. Feature engineering  — cutoff_date prevents look-ahead bias [WF-1]
B. ML alpha             — per-asset IC; only IC > 0 views forwarded to BL
C. Volatility (GARCH)   — per-asset regime; majority-vote portfolio regime
D. BL optimisation      — MV (low-vol) or Min-CVaR (high-vol) with ML views
E. Ablation             — BL-only (no ML views) for attribution analysis

Outputs
-------
A dict containing:
  - backtester         : PortfolioBacktester instance (fitted)
  - metrics            : strategy performance dict
  - ablation_metrics   : BL-only performance dict
  - weights_history    : pd.DataFrame of full-period weight snapshots
  - ml_metrics         : per-asset IC / R² / calibration
  - ic_history         : per-asset IC time series across rebalances
  - spy_returns        : SPY daily return series (benchmark)
  - prices             : full price DataFrame
  - common_tickers     : tickers with complete price data
  - port_rets          : daily strategy return series
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest import PortfolioBacktester, StatisticalTests
from src.config import Config, DEFAULT_CONFIG
from src.data import DataEngine
from src.ml_alpha import MLAlphaEngine
from src.optimizer import BlackLittermanOptimizer
from src.volatility import VolatilityEngine

log = logging.getLogger(__name__)


def run_walk_forward(cfg: Config = None) -> Dict:
    """
    Execute the full walk-forward Alpha Engine pipeline.

    Parameters
    ----------
    cfg : Config, optional
        Override any parameters via Config / sub-configs.
        Defaults to DEFAULT_CONFIG (see src/config.py).

    Returns
    -------
    dict
        All artefacts from the run (backtester, metrics, weights, etc.).
        See module docstring for full key list.
    """
    cfg = cfg or DEFAULT_CONFIG
    bc = cfg.backtest
    ml_cfg = cfg.ml
    opt_cfg = cfg.optimizer
    universe: List[str] = cfg.universe

    print(f"\n{'='*70}")
    print("  WALK-FORWARD ALPHA ENGINE")
    print(f"{'='*70}")
    print(f"Universe: {len(universe)} assets | Window: {bc.train_window}d | Freq: {bc.rebalance_freq}")

    # ── Stage 0: Data ──────────────────────────────────────────────────────
    data_eng = DataEngine(bc.start_date, bc.end_date)
    prices = data_eng.fetch_universe(universe)
    market_caps = data_eng.fetch_market_caps(universe)
    returns = prices.pct_change().dropna()
    print(f"Loaded: {len(prices)} trading days, {prices.shape[1]} assets")

    # ── Rebalance schedule (aligned to actual trading days) ────────────────
    reb_raw = pd.date_range(prices.index[bc.train_window], prices.index[-1], freq=bc.rebalance_freq)
    reb_dates = pd.DatetimeIndex(sorted(set(
        prices.index[min(prices.index.searchsorted(d), len(prices) - 1)]
        for d in reb_raw
        if prices.index.searchsorted(d) >= bc.train_window
    )))
    print(f"Rebalance schedule: {len(reb_dates)} periods")

    # ── Walk-forward loop ──────────────────────────────────────────────────
    weights_record: Dict = {}
    ablation_record: Dict = {}
    ml_metrics_all: Dict = {}
    ic_history: Dict[str, list] = {t: [] for t in universe}

    for i, reb_date in enumerate(reb_dates):
        loc = prices.index.get_loc(reb_date)
        start_loc = max(0, loc - bc.train_window)

        train_prices = prices.iloc[start_loc:loc]
        train_returns = returns.iloc[start_loc:loc]
        n_obs = len(train_returns)

        if n_obs < 252:
            continue

        print(f"  [{i+1}/{len(reb_dates)}] {reb_date.date()} (n={n_obs})", end="  ")

        # A. Features — restrict to data available up to reb_date [WF-1]
        data_eng.data["prices"] = train_prices
        feat_dict = data_eng.engineer_features(cutoff_date=str(reb_date.date()))
        data_eng.data["prices"] = prices  # restore full price history

        # B. ML alpha — train per asset, keep views where IC > 0
        ml_eng = MLAlphaEngine(cfg=ml_cfg)
        ml_views: Dict[str, float] = {}

        for t in universe:
            if t not in feat_dict:
                continue
            X, y = ml_eng.prepare_data(feat_dict[t].dropna(), horizon=ml_cfg.horizon)
            if len(X) < ml_cfg.min_train_obs:
                continue
            try:
                m = ml_eng.train(X, y, t)
                ic_history[t].append(m["IC"])
                if m["IC"] > 0:
                    ml_views[t] = float(ml_eng.predict(X.iloc[-1:], t)[0])
                    ml_metrics_all[t] = m
            except Exception:
                pass

        # C. GJR-GARCH vol forecasts and regime detection
        vol_eng = VolatilityEngine()
        regimes: Dict[str, str] = {}

        for t in list(ml_views.keys()):
            if t not in train_returns.columns:
                continue
            try:
                if vol_eng.fit(train_returns[t], t) is None:
                    continue
                vol_eng.forecast(t, horizon=5)
                regimes[t] = vol_eng.regime(t, pct=opt_cfg.regime_vol_pct)
            except Exception:
                pass

        if regimes:
            n_high = sum(r == "high_vol" for r in regimes.values())
            portfolio_regime = "high_vol" if n_high > len(regimes) / 2 else "low_vol"
        else:
            portfolio_regime = "low_vol"

        # D. BL optimisation
        selected = [t for t in ml_views if t in train_returns.columns]

        def _optimize(use_ml_views: bool) -> pd.Series:
            if len(selected) < 3:
                return pd.Series(1 / len(universe), index=universe)
            try:
                bl = BlackLittermanOptimizer(cfg=opt_cfg)
                tr = train_returns[selected].dropna()
                mc = market_caps.reindex(selected).fillna(market_caps.median())
                eq, cov = bl.equilibrium(tr, mc)

                mu = (
                    bl.posterior(eq, cov, ml_views, n_obs=n_obs)
                    if use_ml_views else eq
                )
                wts = (
                    bl.optimize_min_cvar(tr, max_weight=opt_cfg.max_weight)
                    if portfolio_regime == "high_vol"
                    else bl.optimize_mv(mu, cov, max_weight=opt_cfg.max_weight,
                                        regime=portfolio_regime)
                )
                wts = wts.reindex(universe).fillna(0)
                return wts / wts.sum() if wts.sum() > 0 else pd.Series(1 / len(universe), index=universe)

            except Exception as exc:
                log.warning("Optimisation failed: %s → equal weights", exc)
                return pd.Series(1 / len(universe), index=universe)

        wts = _optimize(use_ml_views=True)
        weights_record[reb_date] = wts
        if bc.run_ablation:
            ablation_record[reb_date] = _optimize(use_ml_views=False)

        top3 = wts[wts > 0.01].nlargest(3)
        print(
            f"regime={portfolio_regime}  views={len(ml_views)}  "
            f"top=[{', '.join(f'{t}:{v:.0%}' for t, v in top3.items())}]"
        )

    if not weights_record:
        raise RuntimeError("No weight snapshots — check data quality and date range.")

    def _assemble(record: Dict) -> pd.DataFrame:
        eq = 1 / len(universe)
        wh = pd.DataFrame(record).T.reindex(prices.index).ffill()
        fvi = wh.first_valid_index()
        if fvi:
            wh.loc[wh.index < fvi] = eq
        return wh.fillna(eq)

    weights_history = _assemble(weights_record)
    common_tickers = [t for t in universe if t in prices.columns]

    # ── Backtest ───────────────────────────────────────────────────────────
    print("\nRunning backtest …")
    bt = PortfolioBacktester(bc.initial_capital, opt_cfg.risk_free_rate)
    ph = bt.run(prices[common_tickers], weights_history[common_tickers], bc.rebalance_freq)
    spy = prices["SPY"].pct_change() if "SPY" in prices.columns else None
    m = bt.metrics(spy)

    print("\n" + "=" * 55)
    print("  PERFORMANCE SUMMARY — ML + BL STRATEGY")
    print("=" * 55)
    for k, v in m.items():
        print(f"  {k:<35} {v}")

    # ── Ablation: BL-only ──────────────────────────────────────────────────
    ablation_metrics: Dict = {}
    if bc.run_ablation and ablation_record:
        abl_wh = _assemble(ablation_record)
        abl_bt = PortfolioBacktester(bc.initial_capital, opt_cfg.risk_free_rate)
        abl_bt.run(prices[common_tickers], abl_wh[common_tickers], bc.rebalance_freq)
        ablation_metrics = abl_bt.metrics(spy)
        print("\n  ABLATION — BL-only (no ML views)")
        for k, v in ablation_metrics.items():
            print(f"  {k:<35} {v:>12}  (ML+BL: {m.get(k, '—'):>12})")

    # ── Statistical validation ─────────────────────────────────────────────
    port_rets = ph["portfolio_value"].pct_change().dropna()
    StatisticalTests.report(ph, ml_metrics_all, n_trials=max(len(ml_metrics_all), 1),
                            rf=opt_cfg.risk_free_rate)

    print("\n  Per-asset IC significance:")
    for t, ics in ic_history.items():
        if len(ics) >= 5:
            r = StatisticalTests.ic_significance(pd.Series(ics))
            if "error" not in r:
                sig = "✓" if r["significant_5pct"] else "✗"
                print(
                    f"    {t:<6} mean_IC={r['mean_IC']:.3f}  "
                    f"ICIR={r['ICIR']:.2f}  p={r['p_value']:.3f}  {sig}"
                )

    return {
        "backtester": bt,
        "metrics": m,
        "ablation_metrics": ablation_metrics,
        "weights_history": weights_history,
        "ml_metrics": ml_metrics_all,
        "ic_history": ic_history,
        "spy_returns": spy,
        "prices": prices,
        "common_tickers": common_tickers,
        "port_rets": port_rets,
    }
