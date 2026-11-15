"""Backtest and the statistics used to judge it.

backtest            daily simulation with monthly rebalancing and trading costs
summarize           return, volatility, Sharpe, drawdown from daily excess returns
sharpe_ci           block bootstrap confidence interval for one Sharpe ratio
sharpe_diff_test    paired block bootstrap for the difference of two Sharpe ratios
deflated_sharpe     Deflated Sharpe Ratio of Bailey and Lopez de Prado (2014)
ic_stats            mean rank IC with a Newey-West t-statistic
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import COMMISSION, DEFAULT_HALF_SPREAD, HALF_SPREAD, SEED

EULER = 0.5772156649015329


def cost_vector(tickers, cost_mult: float = 1.0) -> np.ndarray:
    """One-way cost per unit traded for each ticker."""
    return cost_mult * np.array([HALF_SPREAD.get(t, DEFAULT_HALF_SPREAD) + COMMISSION for t in tickers])


def backtest(prices: pd.DataFrame, targets: pd.DataFrame, cost_mult: float = 1.0) -> Dict:
    """Simulate a fully invested portfolio that trades to targets at the close of each target date.

    prices   daily total return indices without gaps
    targets  weights per rebalance date (rows sum to 1); dates must be in prices.index

    Between rebalances the weights drift with returns. On a rebalance date the
    day's return is earned on the old weights, then the portfolio trades and
    pays cost * |trade| out of its value. The first target date only sets up
    the portfolio, so returns start on the following day.

    Returns daily net returns, one-way turnover per rebalance and total cost paid.
    """
    targets = targets.reindex(columns=prices.columns).fillna(0.0)
    start = targets.index[0]
    px = prices.loc[start:]
    rets = px.pct_change().fillna(0.0).values
    cost = cost_vector(prices.columns, cost_mult)
    tgt = {d: targets.loc[d].values for d in targets.index}
    w = np.zeros(px.shape[1])
    out = np.zeros(len(px))
    turnover = {}
    for i, d in enumerate(px.index):
        gross = 1.0 + float(w @ rets[i])
        if gross > 0 and w.sum() > 0:
            w = w * (1.0 + rets[i]) / gross
        day = gross - 1.0
        if d in tgt:
            trade = np.abs(tgt[d] - w)
            # setting up the initial portfolio is not charged, for any strategy
            paid = float(trade @ cost) if i > 0 else 0.0
            turnover[d] = 0.5 * float(trade.sum()) if i > 0 else np.nan
            day = (1.0 + day) * (1.0 - paid) - 1.0
            w = tgt[d].copy()
        out[i] = day
    r = pd.Series(out, index=px.index).iloc[1:]
    to = pd.Series(turnover).dropna()
    return {"returns": r, "turnover": to}


def sharpe(excess: np.ndarray) -> float:
    sd = excess.std(ddof=1)
    return float(np.sqrt(252) * excess.mean() / sd) if sd > 0 else np.nan


def summarize(ret: pd.Series, rf: pd.Series, turnover: Optional[pd.Series] = None) -> Dict[str, float]:
    """Annualised statistics of a daily return series, in excess of the daily bill rate."""
    ret = ret.dropna()
    ex = ret - rf.reindex(ret.index).fillna(0.0)
    cum = (1 + ret).cumprod()
    out = {
        "ann_return": float(ret.mean() * 252),
        "ann_excess": float(ex.mean() * 252),
        "ann_vol": float(ret.std() * np.sqrt(252)),
        "sharpe": sharpe(ex.values),
        "max_drawdown": float((cum / cum.cummax() - 1).min()),
        "n_days": int(len(ret)),
    }
    if turnover is not None and len(turnover):
        out["turnover_ann"] = float(turnover.mean() * 12)
    return out


def _block_index(n: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    """Circular block bootstrap index matrix of shape (n_boot, n)."""
    rng = np.random.default_rng(seed)
    k = n // block + 1
    starts = rng.integers(0, n, size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, -1)[:, :n]


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    return np.sqrt(252) * x.mean(axis=1) / x.std(axis=1, ddof=1)


def sharpe_ci(excess: pd.Series, n_boot: int = 2000, block: int = 21, seed: int = SEED) -> Dict:
    """Sharpe ratio with a 95% circular block bootstrap interval."""
    x = excess.dropna().values
    boot = _sharpe_rows(x[_block_index(len(x), block, n_boot, seed)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"sharpe": sharpe(x), "lo": float(lo), "hi": float(hi)}


def sharpe_diff_test(a: pd.Series, b: pd.Series, n_boot: int = 2000, block: int = 21,
                     seed: int = SEED) -> Dict:
    """Difference of Sharpe ratios (a minus b) with a paired block bootstrap.

    Both series are resampled with the same blocks, which keeps their
    correlation. The p-value is two-sided for H0: equal Sharpe ratios, from the
    bootstrap distribution centred on the observed difference.
    """
    df = pd.concat([a, b], axis=1).dropna()
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    obs = sharpe(x) - sharpe(y)
    idx = _block_index(len(df), block, n_boot, seed)
    d = _sharpe_rows(x[idx]) - _sharpe_rows(y[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(np.mean(np.abs(d - obs) >= abs(obs)))
    return {"diff": float(obs), "lo": float(lo), "hi": float(hi), "p_value": p}


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float,
                    n_trials: int = 1, trials_sr_std: float = 0.0) -> Dict:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014).

    All inputs are per period, not annualised: sr is mean / standard deviation
    of the n_obs returns, kurt is the raw kurtosis (3 for a normal).

    The benchmark SR0 is the Sharpe ratio expected from the best of n_trials
    strategies that all have zero true Sharpe, given the standard deviation of
    Sharpe ratios across the trials. The result is the probability that the true
    Sharpe ratio exceeds SR0. With n_trials = 1 it is the Probabilistic Sharpe
    Ratio against zero.
    """
    if n_trials > 1:
        sr0 = trials_sr_std * ((1 - EULER) * stats.norm.ppf(1 - 1 / n_trials)
                               + EULER * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    else:
        sr0 = 0.0
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return {"dsr": float(stats.norm.cdf(z)), "sr0": float(sr0)}


def monthly(ret: pd.Series) -> pd.Series:
    return (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1


def rank_ic(signal: pd.Series, outcome: pd.Series, min_assets: int = 5) -> pd.Series:
    """Spearman correlation across assets on each date. Both inputs are indexed by (date, ticker)."""
    df = pd.concat([signal.rename("s"), outcome.rename("o")], axis=1).dropna()
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) >= min_assets:
            out[d] = stats.spearmanr(g["s"], g["o"]).statistic
    return pd.Series(out, dtype=float)


def nw_tstat(x: np.ndarray, lags: int) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) standard error."""
    x = np.asarray(x, float)
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(v / n))


def ic_stats(ic: pd.Series, lags: int = 3) -> Dict:
    """Mean IC, Newey-West t-statistic and p-value, share of positive months."""
    x = ic.dropna().values
    t = nw_tstat(x, lags)
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "t_stat": t, "p_value": float(2 * (1 - stats.norm.cdf(abs(t)))),
            "share_positive": float((x > 0).mean()),
            # smallest mean IC that would give |t| = 2 with this many months and this dispersion
            "detectable_ic": float(2 * x.std(ddof=1) / np.sqrt(len(x)))}
