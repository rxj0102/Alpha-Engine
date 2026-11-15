"""Simple, economically motivated signals and the mapping from a score to an expected return.

All signals are read with a one day lag: the value used for a trade at the close
of day t is computed from data up to day t-1.
"""
from typing import List

import numpy as np
import pandas as pd

from src.opendata import par_bond_duration


def realised_vol(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised volatility of daily returns over a trailing window."""
    return prices.pct_change().rolling(window).std() * np.sqrt(252)


def momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """12-1 momentum: return from 252 to 21 days ago, divided by one year volatility.

    Skipping the latest month and scaling by risk are the usual conventions for
    comparing assets with very different volatility (Asness, Moskowitz and Pedersen 2013).
    """
    ret = prices.shift(21) / prices.shift(252) - 1
    return ret / realised_vol(prices, 252)


def bond_carry(yields: pd.DataFrame, rf_daily: pd.Series) -> pd.DataFrame:
    """Annual carry plus roll-down, in excess of the bill rate, for the three bond series.

    carry = yield - bill rate + duration * slope, where slope is the fall in
    yield per year as the bond ages along today's curve (Koijen et al. 2018).
    Yields are FRED constant maturity series in percent. The Baa series has no
    curve of its own, so it uses the Treasury slope between 10 and 20 years.
    """
    y = yields / 100.0
    bill = rf_daily * 252
    slope_10 = (y["DGS10"] - y["DGS7"]) / 3.0
    slope_20 = (y["DGS20"] - y["DGS10"]) / 10.0
    out = pd.DataFrame(index=yields.index)
    out["IEF"] = y["DGS10"] - bill + par_bond_duration(y["DGS10"], 10) * slope_10
    out["TLT"] = y["DGS20"] - bill + par_bond_duration(y["DGS20"], 20) * slope_20
    out["LQD"] = y["DBAA"] - bill + par_bond_duration(y["DBAA"], 12) * slope_20
    return out


def xs_zscore(row: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score of one date, clipped. Missing values stay missing."""
    x = row.dropna()
    if len(x) < 3 or x.std() == 0:
        return pd.Series(np.nan, index=row.index)
    return ((x - x.mean()) / x.std()).clip(-clip, clip).reindex(row.index)


def ts_zscore(history: pd.DataFrame, min_obs: int, clip: float = 3.0) -> pd.Series:
    """z-score of the last row against the column's own history up to that row."""
    last = history.iloc[-1]
    n = history.notna().sum()
    z = ((last - history.mean()) / history.std()).clip(-clip, clip)
    return z.where(n >= min_obs)


def alpha_from_z(z: pd.Series, vol_ann: pd.Series, ic: float) -> pd.Series:
    """Annual expected excess return implied by a score: alpha = IC * vol * z (Grinold 1994).

    With a one month horizon the monthly alpha is IC * (vol / sqrt(12)) * z;
    multiplying by 12 gives IC * vol * sqrt(12) * z per year.
    """
    return (ic * vol_ann * np.sqrt(12.0) * z).dropna()


def combine(scores: List[pd.Series]) -> pd.Series:
    """Average the scores that exist for each asset."""
    return pd.concat(scores, axis=1).mean(axis=1, skipna=True)
