"""
data.py — Market data fetching and feature engineering.

DataEngine handles:
  - Per-ticker price download via yfinance with robust column extraction
  - Market-cap retrieval for Black-Litterman prior weights
  - Walk-forward feature engineering (momentum, volatility, MACD, RSI, moments)

[WF-1] All features are computed with an explicit cutoff_date so the
walk-forward loop never introduces look-ahead bias.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class DataEngine:
    """Fetches prices, market caps, and engineers per-asset features."""

    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.data: Dict = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_close(raw: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
        """Robustly extract Close prices from yfinance's flat or MultiIndex columns."""
        if raw is None or raw.empty:
            return None
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            s = raw.get(("Close", ticker),
                        raw["Close"].iloc[:, 0] if "Close" in cols.get_level_values(0) else None)
        else:
            s = raw.get("Close") or raw.get("Adj Close") or raw.iloc[:, 0]
        s = s.squeeze() if isinstance(s, pd.DataFrame) else s
        return s if isinstance(s, pd.Series) and not s.empty else None

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_universe(self, tickers: List[str]) -> pd.DataFrame:
        """
        Download adjusted close prices for each ticker individually.

        Per-ticker download avoids bulk-download failures; ffill(limit=5)
        patches short holiday gaps without injecting stale data.

        Returns
        -------
        pd.DataFrame
            Date-indexed price matrix, columns = tickers.
        """
        log.info("Fetching %d assets …", len(tickers))
        prices = {}
        for t in tickers:
            try:
                raw = yf.download(
                    t, start=self.start_date, end=self.end_date,
                    progress=False, auto_adjust=True,
                )
                s = self._extract_close(raw, t)
                if s is not None:
                    prices[t] = s
                    log.info("  ✓ %s: %d obs", t, s.notna().sum())
                else:
                    log.warning("  ✗ %s: no data returned", t)
            except Exception as exc:
                log.warning("  ✗ %s: %s", t, exc)

        if not prices:
            raise RuntimeError("No price data fetched — check tickers and date range.")

        df = pd.DataFrame(prices).dropna(how="all").ffill(limit=5)
        self.data["prices"] = df
        return df

    def fetch_market_caps(self, tickers: List[str]) -> pd.Series:
        """
        Retrieve market cap / AUM for each ticker as Black-Litterman prior weights.

        Falls back to equal weights when data is unavailable. ETF market caps
        serve as a reasonable proxy for relative asset-class size.

        Returns
        -------
        pd.Series
            Raw (un-normalised) market cap per ticker.
        """
        caps: Dict[str, float] = {}
        for t in tickers:
            try:
                info = yf.Ticker(t).fast_info
                mc = getattr(info, "market_cap", None)
                if mc is None or np.isnan(float(mc)):
                    lp = getattr(info, "last_price", None)
                    sh = getattr(info, "shares", None)
                    mc = lp * sh if (lp and sh) else None
                if mc:
                    caps[t] = float(mc)
            except Exception:
                pass

        if not caps:
            return pd.Series(1.0, index=tickers)

        s = pd.Series(caps)
        return s.reindex(tickers).fillna(s.median())

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
            f["bb_width"] = 2 * r.rolling(20).std() / (sma20 / p)

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
