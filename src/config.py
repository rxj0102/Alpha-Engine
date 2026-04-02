"""
config.py — Centralized configuration for the Alpha Engine.

All model parameters, universe definitions, and cost assumptions live here.
Edit this file to run experiments without touching any model code.
"""

from dataclasses import dataclass, field
from typing import List

# ── Random seed ────────────────────────────────────────────────────────────────
SEED: int = 42

# ── Asset universe ─────────────────────────────────────────────────────────────
UNIVERSE: List[str] = [
    # Broad equities
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XLI",
    # Fixed income
    "TLT", "IEF", "LQD",
    # Commodities
    "GLD", "USO",
]

# ── Transaction cost model ─────────────────────────────────────────────────────
# Half bid-ask spread per ticker (bps → decimal). Illiquid names carry wider spreads.
BID_ASK_SPREAD: dict = {
    "SPY": 3e-4, "QQQ": 3e-4, "IWM": 5e-4, "EFA": 5e-4, "EEM": 8e-4,
    "TLT": 4e-4, "IEF": 4e-4, "LQD": 6e-4, "GLD": 4e-4, "USO": 1e-3,
}
DEFAULT_SPREAD: float = 8e-4   # fallback for unlisted tickers
COMMISSION: float = 5e-4       # 5 bps round-trip commission


@dataclass
class BacktestConfig:
    """Parameters controlling the walk-forward backtest."""
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1_000_000
    train_window: int = 504        # ~2 years of trading days
    rebalance_freq: str = "ME"     # month-end rebalancing
    run_ablation: bool = True      # compare ML+BL vs BL-only


@dataclass
class MLConfig:
    """ML ensemble and feature-engineering parameters."""
    horizon: int = 21              # forward-return prediction horizon (trading days)
    n_cv_splits: int = 5           # TimeSeriesSplit folds
    min_train_obs: int = 100       # minimum samples required to train
    rf_n_estimators: int = 200
    rf_max_depth: int = 8
    rf_min_samples_split: int = 15
    rf_max_features: float = 0.5
    gbr_n_estimators: int = 150
    gbr_max_depth: int = 4
    gbr_learning_rate: float = 0.05
    gbr_subsample: float = 0.8
    svr_c: float = 0.5
    svr_epsilon: float = 0.05
    ridge_alpha: float = 1.0
    meta_ridge_alpha: float = 0.5


@dataclass
class OptimizerConfig:
    """Portfolio optimisation parameters."""
    risk_aversion: float = 2.5     # λ in MV objective
    risk_free_rate: float = 0.04   # annualised
    max_weight: float = 0.30       # per-asset upper bound
    view_confidence: float = 0.65  # BL view confidence ∈ (0, 1)
    cvar_alpha: float = 0.05       # CVaR tail probability
    regime_vol_pct: float = 75.0   # percentile threshold for high-vol regime


@dataclass
class Config:
    """Top-level config object; compose sub-configs here."""
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    universe: List[str] = field(default_factory=lambda: UNIVERSE)


# Default config instance — import and override fields as needed.
DEFAULT_CONFIG = Config()
