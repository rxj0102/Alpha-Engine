"""Configuration. Every choice here was fixed before the long-sample run and none was tuned on its results."""
from dataclasses import dataclass, field
from typing import Dict, List

SEED: int = 42

# Labels keep the ETF tickers for readability; the data are open series (see src/opendata.py).
UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "EFA",          # broad equity
    "XLF", "XLE", "XLK", "XLV", "XLI",   # US sectors
    "TLT", "IEF", "LQD",                 # bonds
    "USO",                               # oil
]
BONDS: List[str] = ["TLT", "IEF", "LQD"]

# One-way cost per unit of notional traded: half bid-ask spread plus commission.
# These are ETF-era assumptions applied to the whole sample.
HALF_SPREAD: Dict[str, float] = {
    "SPY": 3e-4, "QQQ": 3e-4, "IWM": 5e-4, "EFA": 5e-4,
    "TLT": 4e-4, "IEF": 4e-4, "LQD": 6e-4, "USO": 1e-3,
}
DEFAULT_HALF_SPREAD: float = 8e-4
COMMISSION: float = 5e-4


@dataclass
class Config:
    start: str = "1990-07-02"        # first day with the Developed ex US series
    end: str = "2026-09-30"          # last day in the Ken French files at download time
    universe: List[str] = field(default_factory=lambda: list(UNIVERSE))
    cov_window: int = 504            # days for the Ledoit-Wolf covariance
    vol_window: int = 63             # days for the volatility used to scale signals
    min_train_months: int = 60       # labelled months required before the first ML forecast
    carry_min_months: int = 36       # history required before a carry z-score is used
    risk_aversion: float = 2.5
    max_weight: float = 0.30
    view_confidence: float = 0.65    # c in the Black-Litterman view uncertainty
    assumed_ic: float = 0.05         # skill assumed when turning a z-score into an expected return
    ridge_alpha: float = 100.0
    gbm_params: Dict = field(default_factory=lambda: dict(
        n_estimators=150, max_depth=3, learning_rate=0.03, subsample=0.8,
        min_samples_leaf=30, random_state=SEED))
    n_bootstrap: int = 2000
    block: int = 21


DEFAULT_CONFIG = Config()
