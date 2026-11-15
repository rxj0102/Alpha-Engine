"""Data loading and features.

Prices are total return indices built from free open series (FRED, Ken French
Data Library) by src/opendata.py. A frozen copy lives in data/snapshot so that
reruns give identical numbers; pass refresh=True to download again.

Every feature is causal: its value on day t uses prices up to and including
day t only. tests/test_data.py checks this by recomputing on truncated data.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src import opendata
from src.config import Config

SNAP = Path(__file__).resolve().parents[1] / "data" / "snapshot"

FEATURES = [
    "ret_1d", "mom_5d", "mom_21d", "mom_63d", "mom_252d",
    "vol_5d", "vol_21d", "vol_63d", "vol_252d",
    "price_to_sma20", "sma20_to_sma50", "macd", "macd_hist", "bb_width", "rsi",
    "ret_mean_63d", "ret_skew_63d", "ret_kurt_63d",
]


def load_panel(cfg: Config, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Return prices, the daily risk-free rate and the yields used for bond carry."""
    files = {k: SNAP / f"{k}.csv" for k in ("prices", "rf", "yields")}
    if refresh or not all(p.exists() for p in files.values()):
        SNAP.mkdir(parents=True, exist_ok=True)
        prices = opendata.proxy_prices(cfg.universe, cfg.start, cfg.end)
        rf = opendata.french("F-F_Research_Data_Factors_daily")["RF"].reindex(prices.index).ffill()
        yields = opendata.fred(["DGS7", "DGS10", "DGS20", "DBAA"]).reindex(prices.index).ffill(limit=5)
        prices.to_csv(files["prices"], float_format="%.6f", index_label="Date")
        rf.to_frame("RF").to_csv(files["rf"], float_format="%.6f", index_label="Date")
        yields.to_csv(files["yields"], float_format="%.4f", index_label="Date")
    read = lambda p: pd.read_csv(p, index_col="Date", parse_dates=True)
    prices = read(files["prices"])[cfg.universe].loc[cfg.start:cfg.end]
    return {"prices": prices,
            "rf": read(files["rf"])["RF"].reindex(prices.index),
            "yields": read(files["yields"]).reindex(prices.index)}


def asset_features(p: pd.Series) -> pd.DataFrame:
    """The 18 technical features for one price series. Scale free and causal."""
    r = p.pct_change()
    f = pd.DataFrame(index=p.index)
    f["ret_1d"] = r
    for n in (5, 21, 63, 252):
        f[f"mom_{n}d"] = p.pct_change(n)
        f[f"vol_{n}d"] = r.rolling(n).std() * np.sqrt(252)
    sma20, sma50 = p.rolling(20).mean(), p.rolling(50).mean()
    f["price_to_sma20"] = p / sma20 - 1
    f["sma20_to_sma50"] = sma20 / sma50 - 1
    macd = p.ewm(span=12, adjust=False).mean() - p.ewm(span=26, adjust=False).mean()
    # divided by price so the value does not depend on the index level
    f["macd"] = macd / p
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / p
    f["bb_width"] = 4 * p.rolling(20).std() / sma20
    gain = r.where(r > 0, 0.0).ewm(com=13, adjust=False).mean()
    loss = (-r.where(r < 0, 0.0)).ewm(com=13, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    f["ret_mean_63d"] = r.rolling(63).mean()
    f["ret_skew_63d"] = r.rolling(63).skew()
    f["ret_kurt_63d"] = r.rolling(63).kurt()
    # ewm features have no built in warm up; blank them until a year of data exists
    f[p.notna().cumsum() < 253] = np.nan
    return f[FEATURES]


def all_features(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {t: asset_features(prices[t]) for t in prices.columns}


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month present in the index."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)
