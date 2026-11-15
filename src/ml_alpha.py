"""Pooled cross-sectional return forecasts.

One model is fitted on every asset and month at once. Each row is an asset on a
month-end; features are ranked across assets on that date; the target is next
month's return, divided by volatility and measured relative to the average
asset. Pooling matters because one asset alone gives 12 non-overlapping monthly
observations a year, far too few for a non-linear model.

Walk-forward protocol for the forecast traded at month-end k:
  features    data up to the day before month-end k
  training    rows 0 .. k-2 (row k-1 is left out because its label, the return
              over month k-1 to k, is only complete at the close we trade on)
  models      ridge and gradient boosting with fixed hyperparameters, no tuning
  ensemble    mean of the two forecasts after a cross-sectional z-score
"""
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.config import Config
from src.data import FEATURES
from src.signals import xs_zscore

MODELS = ["ridge", "gbm", "ensemble"]


def build_panel(features: Dict[str, pd.DataFrame], prices: pd.DataFrame,
                vol: pd.DataFrame, reb: pd.DatetimeIndex):
    """Return X (ranked features), y (target) and fwd (vol-scaled next month return).

    All three are indexed by (rebalance date, ticker). X uses the day before the
    rebalance date. y is missing on the last date, which has no next month yet.
    """
    idx = prices.index
    loc = idx.get_indexer(reb)
    prev = idx[loc - 1]
    xs, fw = [], []
    px = prices.ffill()
    for k, d in enumerate(reb):
        x = pd.DataFrame({t: features[t].loc[prev[k]] for t in prices.columns}).T
        x = x.dropna()
        x = x.rank() / len(x) - 0.5 - 0.5 / len(x)      # centred ranks in (-0.5, 0.5)
        x.index = pd.MultiIndex.from_product([[d], x.index], names=["date", "ticker"])
        xs.append(x)
        if k + 1 < len(reb):
            r = px.iloc[loc[k + 1]] / px.iloc[loc[k]] - 1
            f = (r / vol.loc[prev[k]]).dropna()
            f.index = pd.MultiIndex.from_product([[d], f.index], names=["date", "ticker"])
            fw.append(f)
    X = pd.concat(xs)[FEATURES]
    fwd = pd.concat(fw).rename("fwd").reindex(X.index)
    y = (fwd - fwd.groupby(level="date").transform("mean")).rename("y")
    return X, y, fwd


def walk_forward(X: pd.DataFrame, y: pd.Series, cfg: Config, verbose: bool = False) -> pd.DataFrame:
    """Out-of-sample forecasts for every rebalance date with enough history.

    Returns a frame indexed like X with columns ridge, gbm and ensemble (each a
    cross-sectional z-score). Dates before the first forecast are absent.
    """
    dates = X.index.get_level_values("date").unique()
    out = []
    for k, d in enumerate(dates):
        train_dates = dates[: max(k - 1, 0)]
        if len(train_dates) < cfg.min_train_months:
            continue
        tr = X.index.get_level_values("date").isin(train_dates) & y.notna().values
        Xt, yt, Xp = X[tr], y[tr], X.xs(d, level="date")
        ridge = Ridge(alpha=cfg.ridge_alpha).fit(Xt, yt)
        gbm = GradientBoostingRegressor(**cfg.gbm_params).fit(Xt, yt)
        f = pd.DataFrame({"ridge": xs_zscore(pd.Series(ridge.predict(Xp), index=Xp.index)),
                          "gbm": xs_zscore(pd.Series(gbm.predict(Xp), index=Xp.index))})
        f["ensemble"] = xs_zscore(f.mean(axis=1))
        f.index = pd.MultiIndex.from_product([[d], f.index], names=["date", "ticker"])
        out.append(f)
        if verbose and len(out) % 60 == 1:
            print(f"  forecast {d.date()}  training rows {int(tr.sum())}")
    return pd.concat(out)
