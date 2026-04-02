"""
Alpha Engine — Systematic multi-asset portfolio construction.

Modules
-------
config          Centralised parameters (backtest, ML, optimizer)
data            DataEngine: price fetching and feature engineering
ml_alpha        MLAlphaEngine: stacked ensemble return forecaster
volatility      VolatilityEngine: GJR-GARCH regime detection
optimizer       BlackLittermanOptimizer: Bayesian portfolio construction
backtest        PortfolioBacktester + StatisticalTests
visualization   Performance and risk decomposition charts
pipeline        run_walk_forward: full orchestration entry point
"""

from src.pipeline import run_walk_forward
from src.config import Config, DEFAULT_CONFIG

__all__ = ["run_walk_forward", "Config", "DEFAULT_CONFIG"]
