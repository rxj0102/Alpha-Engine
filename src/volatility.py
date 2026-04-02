"""
volatility.py — GJR-GARCH(1,1,1) volatility forecasting with leverage effect.

Why GJR-GARCH and not standard GARCH?
--------------------------------------
Standard GARCH(1,1) treats upside and downside shocks symmetrically:
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

This is empirically wrong for equities. Black (1976) and Christie (1982)
documented that equity vol rises more sharply after negative returns than
after positive returns of equal magnitude — the "leverage effect". GJR-GARCH
captures this with an asymmetry indicator:

    σ²_t = ω + (α + γ·I[ε_{t-1}<0])·ε²_{t-1} + β·σ²_{t-1}

For a negative shock: effective ARCH coefficient = α + γ
For a positive shock: effective ARCH coefficient = α
Empirically γ ≈ 0.05–0.15 for SPY, meaning negative shocks have roughly
1.5–2× the vol impact of equivalent positive shocks.

Key model properties
--------------------
Persistence  = α + γ/2 + β
    Measures the rate of mean-reversion. For equities, persistence ≈ 0.97,
    implying a half-life of vol shocks of ~23 days (= log(0.5)/log(0.97)).
    Values above 0.999 indicate near-unit-root vol (IGARCH regime).

Long-run variance  = ω / (1 - persistence)
    The unconditional variance to which conditional vol reverts.
    Annualised: √(long_run_var) / 100 * √252.

Distribution: Skewed Student-t
    The skewness parameter accommodates negative return skew (common in
    equity ETFs). The degrees-of-freedom parameter captures excess kurtosis.
    Using Gaussian errors underestimates tail risk in the GARCH likelihood.

Regime detection
----------------
Binary regime (high_vol / low_vol) is based on the 75th percentile of the
in-sample conditional vol history. Using each asset's own history makes
the threshold adaptive across assets with very different base volatility
levels (e.g. USO vs. IEF). The portfolio regime is the majority vote across
all assets with ML views.
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
