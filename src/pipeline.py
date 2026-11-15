"""The experiment: one portfolio construction, several sources of views.

Every strategy uses the same universe, month-end rebalance dates, covariance,
Black-Litterman prior, optimiser and trading costs. Only the views differ:

  1/N                 no views (this is the prior portfolio)
  Inverse volatility  no forecasts, weights proportional to 1 / volatility
  Momentum            12-1 risk-adjusted momentum, ranked across the 13 assets
  Carry (bonds)       bond carry and roll-down against its own history, 3 bond assets only
  Momentum + carry    average of the two scores
  ML                  pooled ridge and gradient boosting ensemble on 18 technical features

A score z becomes a view through alpha = IC * vol * z with the same assumed IC
for every signal, so no signal is given more weight than another by construction.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import backtest as bt
from src import ml_alpha, optimizer, signals
from src.config import BONDS, Config, DEFAULT_CONFIG
from src.data import all_features, load_panel, month_ends

FORECAST_STRATEGIES = ["Momentum", "Carry (bonds)", "Momentum + carry", "ML"]
STRATEGIES = ["1/N", "Inverse volatility"] + FORECAST_STRATEGIES
MARKET = "US market"
CLASSES = {"equity": ["SPY", "QQQ", "IWM", "EFA", "XLF", "XLE", "XLK", "XLV", "XLI"],
           "bonds": BONDS, "oil": ["USO"]}


def build_scores(cfg: Config, panel: Dict, verbose: bool = False) -> Dict:
    """Signals and forecasts for every rebalance date, all lagged one day."""
    prices = panel["prices"]
    idx = prices.index
    vol = signals.realised_vol(prices, cfg.vol_window)
    reb = month_ends(idx)
    reb = reb[idx.get_indexer(reb) >= 260]
    prev = idx[idx.get_indexer(reb) - 1]

    X, y, fwd = ml_alpha.build_panel(all_features(prices), prices, vol, reb)
    fc = ml_alpha.walk_forward(X, y, cfg, verbose=verbose)

    mom = signals.momentum(prices).loc[prev]
    mom.index = reb
    z_mom = mom.apply(signals.xs_zscore, axis=1)

    carry = signals.bond_carry(panel["yields"], panel["rf"]).loc[prev]
    carry.index = reb
    z_carry = pd.DataFrame({d: signals.ts_zscore(carry.iloc[: k + 1], cfg.carry_min_months)
                            for k, d in enumerate(reb)}).T[BONDS]

    first_ml = fc.index.get_level_values("date").min()
    first_carry = z_carry.dropna().index.min()
    eval_start = max(first_ml, first_carry)
    return {"reb": reb, "prev": prev, "vol": vol, "X": X, "y": y, "fwd": fwd, "forecasts": fc,
            "z_mom": z_mom, "carry": carry, "z_carry": z_carry, "eval_start": eval_start}


def scores_for(name: str, d: pd.Timestamp, sc: Dict) -> Optional[pd.Series]:
    """The z-score vector a strategy uses on date d (None for strategies without views)."""
    if name == "Momentum":
        return sc["z_mom"].loc[d].dropna()
    if name == "Carry (bonds)":
        return sc["z_carry"].loc[d].dropna()
    if name == "Momentum + carry":
        return signals.combine([sc["z_mom"].loc[d], sc["z_carry"].loc[d]]).dropna()
    if name == "ML":
        return sc["forecasts"]["ensemble"].xs(d, level="date").dropna()
    return None


def build_weights(cfg: Config, panel: Dict, sc: Dict, assumed_ic: Optional[float] = None,
                  names: Optional[List[str]] = None, cache: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """Target weights per strategy on every rebalance date from eval_start."""
    prices = panel["prices"]
    rets = prices.pct_change()
    idx = prices.index
    ic = cfg.assumed_ic if assumed_ic is None else assumed_ic
    names = names or STRATEGIES
    cache = {} if cache is None else cache
    out = {n: {} for n in names}
    for d, p in zip(sc["reb"], sc["prev"]):
        if d < sc["eval_start"]:
            continue
        if d not in cache:
            loc = idx.get_loc(d)
            cov = optimizer.ledoit_wolf_cov(rets.iloc[loc - cfg.cov_window: loc])
            cache[d] = (cov, optimizer.equilibrium(cov, cfg.risk_aversion))
        cov, pi = cache[d]
        v = sc["vol"].loc[p]
        for n in names:
            if n == "1/N":
                w = pd.Series(1.0 / len(pi), index=pi.index)
            elif n == "Inverse volatility":
                w = (1.0 / v) / (1.0 / v).sum()
            else:
                alpha = signals.alpha_from_z(scores_for(n, d, sc), v, ic)
                mu = optimizer.posterior(pi, cov, pi.reindex(alpha.index) + alpha, cfg.view_confidence)
                w = optimizer.optimize_mv(mu, cov, cfg.risk_aversion, cfg.max_weight)
            out[n][d] = w
    return {n: pd.DataFrame(w).T[list(prices.columns)] for n, w in out.items()}


def run_experiment(cfg: Config = None, refresh: bool = False, verbose: bool = True) -> Dict:
    cfg = cfg or DEFAULT_CONFIG
    panel = load_panel(cfg, refresh=refresh)
    prices, rf = panel["prices"], panel["rf"]
    if verbose:
        print(f"data {prices.index[0].date()} to {prices.index[-1].date()}, {prices.shape[1]} assets")
    sc = build_scores(cfg, panel, verbose=verbose)
    if verbose:
        print(f"first out-of-sample rebalance {sc['eval_start'].date()}")
    cache: Dict = {}
    weights = build_weights(cfg, panel, sc, cache=cache)
    px = prices.ffill()
    runs = {n: bt.backtest(px, w) for n, w in weights.items()}
    returns = pd.DataFrame({n: r["returns"] for n, r in runs.items()})
    returns[MARKET] = px["SPY"].pct_change().reindex(returns.index)

    # robustness: trading costs and the assumed skill level
    robust = {}
    for label, mult in (("no costs", 0.0), ("double costs", 2.0)):
        robust[label] = {n: bt.backtest(px, weights[n], cost_mult=mult)["returns"]
                         for n in ["1/N"] + FORECAST_STRATEGIES}
    for label, ic in (("assumed IC 0.025", 0.025), ("assumed IC 0.10", 0.10)):
        w2 = build_weights(cfg, panel, sc, assumed_ic=ic, names=FORECAST_STRATEGIES, cache=cache)
        robust[label] = {n: bt.backtest(px, w2[n])["returns"] for n in FORECAST_STRATEGIES}

    return {"cfg": cfg, "panel": panel, "scores": sc, "weights": weights, "returns": returns,
            "turnover": {n: r["turnover"] for n, r in runs.items()}, "robust": robust, "rf": rf}
