"""
data.py — Market data loading and feature engineering.

DataEngine handles:
  - Daily returns for open data stand ins of each ETF (FRED, Ken French
    Data Library), converted to total return indices. No Yahoo Finance.
  - Equal prior weights for Black-Litterman (no open market cap source)
  - Walk-forward feature engineering (momentum, volatility, MACD, RSI, moments)

[WF-1] All features are computed with an explicit cutoff_date so the
walk-forward loop never introduces look-ahead bias.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.opendata import PROXY_DESCRIPTIONS, proxy_prices

log = logging.getLogger(__name__)


class DataEngine:
    """Loads open data prices, prior weights, and engineers per-asset features."""

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.data: Dict = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_universe(self, tickers: List[str]) -> pd.DataFrame:
        """
        Build a total return index for each ticker from its open data stand in.

        See src/opendata.py PROXY_DESCRIPTIONS for the series behind each
        ticker. Tickers without an open daily source are dropped.

        Returns
        -------
        pd.DataFrame
            Date-indexed price matrix, columns = tickers.
        """
        log.info("Loading %d assets from open data …", len(tickers))
        df = proxy_prices(tickers, self.start_date, self.end_date)
        for t in df.columns:
            log.info("  %s: %d obs (%s)", t, df[t].notna().sum(), PROXY_DESCRIPTIONS.get(t, ""))
        if df.empty:
            raise RuntimeError("No price data loaded; check tickers and date range.")
        df = df.dropna(how="all").ffill(limit=5)
        self.data["prices"] = df
        return df

    def fetch_market_caps(self, tickers: List[str]) -> pd.Series:
        """
        Black-Litterman prior weights.

        There is no free open source for ETF AUM, so the prior is equal
        weight across the tickers that have data.
        """
        return pd.Series(1.0, index=tickers)

    def engineer_features(self, cutoff_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Compute an 18-factor feature set per asset.

        Factors
        -------
        - Momentum        : 5 / 21 / 63 / 252-day returns
        - Realised vol    : 5 / 21 / 63 / 252-day annualised std
        - Trend           : price-to-SMA20, SMA20-to-SMA50
        - MACD            : signal line and histogram
        - Bollinger width : 2σ / SMA20 normalised spread
        - Wilder RSI      : com=13 ≡ span=27
        - Rolling moments : 63d mean, skewness, excess kurtosis

        Parameters
        ----------
        cutoff_date : str, optional
            [WF-1] Upper date bound. When provided (inside the walk-forward loop)
            features are built only from data available at that date,
            eliminating any look-ahead bias.

        Returns
        -------
        dict[str, pd.DataFrame]
            Ticker → DataFrame of features aligned to price dates.
        """
        prices = self.data["prices"]
        if cutoff_date:
            prices = prices.loc[:cutoff_date]

        returns = prices.pct_change()
        features: Dict[str, pd.DataFrame] = {}

        for t in prices.columns:
            p, r = prices[t], returns[t]
            f = pd.DataFrame(index=prices.index)
            f["price"] = p
            f["return_1d"] = r

            for period in [5, 21, 63, 252]:
                f[f"mom_{period}d"] = p.pct_change(period)
                f[f"vol_{period}d"] = r.rolling(period).std() * np.sqrt(252)

            sma20 = p.rolling(20).mean()
            sma50 = p.rolling(50).mean()
            f["price_to_sma20"] = p / sma20 - 1
            f["sma20_to_sma50"] = sma20 / sma50 - 1

            ema12 = p.ewm(span=12, adjust=False).mean()
            ema26 = p.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            f["macd"] = macd
            f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
            f["bb_width"] = 4 * p.rolling(20).std() / sma20

            # Wilder RSI: com=13 ≡ span=27, matching Wilder's original smoothing constant
            gain = r.where(r > 0, 0.0).ewm(com=13, adjust=False).mean()
            loss = (-r.where(r < 0, 0.0)).ewm(com=13, adjust=False).mean()
            f["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

            f["ret_mean_63d"] = r.rolling(63).mean()
            f["ret_skew_63d"] = r.rolling(63).skew()
            f["ret_kurt_63d"] = r.rolling(63).kurt()

            features[t] = f

        self.data["features"] = features
        return features
