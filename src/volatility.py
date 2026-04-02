"""
volatility.py — GJR-GARCH(1,1,1) volatility forecasting with leverage effect.

Model
-----
σ²_t = ω + (α + γ·I[ε_{t-1}<0])·ε²_{t-1} + β·σ²_{t-1}

γ > 0 captures the leverage effect: equity volatility rises more sharply
in response to negative news than to equivalent positive news (Black 1976).

Distribution: Skewed Student-t to accommodate fat tails and asymmetry in
daily equity returns — more realistic than Gaussian GARCH.

Regime detection: A per-asset binary regime (high_vol / low_vol) is derived
by comparing the current forecast against its own historical percentile.
The portfolio-level regime aggregates individual asset regimes by majority vote.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from arch import arch_model

log = logging.getLogger(__name__)


class VolatilityEngine:
    """
    GJR-GARCH(1,1,1) with skewed-t errors for volatility forecasting.

    Usage
    -----
    vol_eng = VolatilityEngine()
    vol_eng.fit(returns, ticker)
    forecast = vol_eng.forecast(ticker, horizon=5)
    regime   = vol_eng.regime(ticker)
    """

    def __init__(self) -> None:
        self.models: Dict = {}
        self.forecasts: Dict = {}
        self._hist_vol: Dict = {}

    def fit(self, returns: pd.Series, ticker: str) -> Optional[Dict]:
        """
        Fit a GJR-GARCH(1,1,1)-skewt model.

        The series is rescaled by ×100 (arch convention) so that the
        conditional variance remains numerically well-conditioned.

        Parameters
        ----------
        returns : pd.Series
            Daily log or arithmetic returns (NaNs dropped internally).
        ticker : str
            Asset identifier.

        Returns
        -------
        dict or None
            Parameter dict with keys: alpha, gamma, beta, omega,
            persistence, long_run_vol_ann. Returns None on fit failure.
        """
        try:
            res = arch_model(
                returns.dropna() * 100,
                vol="GARCH", p=1, o=1, q=1,
                dist="skewt", rescale=False,
            ).fit(disp="off", show_warning=False)

            self.models[ticker] = res
            self._hist_vol[ticker] = np.sqrt(res.conditional_volatility) / 100 * np.sqrt(252)

            a = res.params.get("alpha[1]", np.nan)
            g = res.params.get("gamma[1]", np.nan)
            b = res.params.get("beta[1]", np.nan)
            o = res.params.get("omega", np.nan)

            # Persistence = α + 0.5γ + β (GJR convention; captures mean-reversion speed)
            persist = a + 0.5 * g + b if not any(np.isnan(v) for v in [a, g, b]) else np.nan
            lr_vol = (
                np.sqrt(o / max(1 - persist, 1e-6)) / 100 * np.sqrt(252)
                if not np.isnan(persist) else np.nan
            )

            params = {
                "alpha": a, "gamma": g, "beta": b, "omega": o,
                "persistence": persist, "long_run_vol_ann": lr_vol,
            }
            log.info(
                "  %s: α=%.4f γ=%.4f β=%.4f persist=%.4f LR_vol=%.2%%",
                ticker, a, g, b, persist, lr_vol,
            )
            return params

        except Exception as exc:
            log.error("  GARCH failed %s: %s", ticker, exc)
            return None

    def forecast(self, ticker: str, horizon: int = 5) -> np.ndarray:
        """
        Return annualised conditional volatility forecast.

        Parameters
        ----------
        ticker : str
            Must have been previously fitted via `fit()`.
        horizon : int
            Forecast horizon in trading days.

        Returns
        -------
        np.ndarray
            Annualised vol for each of the next `horizon` days.
        """
        fc = self.models[ticker].forecast(horizon=horizon, reindex=False)
        vol_ann = np.sqrt(fc.variance.values[-1]) / 100 * np.sqrt(252)
        self.forecasts[ticker] = vol_ann
        return vol_ann

    def regime(self, ticker: str, pct: float = 75.0) -> str:
        """
        Classify the current vol environment as 'high_vol' or 'low_vol'.

        Compares the mean near-term forecast against the `pct`-th percentile
        of the in-sample conditional vol history. Using a rolling historical
        reference rather than a fixed threshold makes the regime detection
        adaptive to each asset's own vol dynamics.

        Parameters
        ----------
        pct : float
            Percentile threshold (default 75 → top quartile = high-vol regime).

        Returns
        -------
        str
            'high_vol' or 'low_vol'
        """
        hist = self._hist_vol[ticker]
        threshold = np.percentile(hist[~np.isnan(hist)], pct)
        current = float(np.mean(self.forecasts[ticker]))
        return "high_vol" if current > threshold else "low_vol"
