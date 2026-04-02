"""
visualization.py — Portfolio performance and risk decomposition charts.

plot_performance        4-panel dashboard: cumulative return, drawdown,
                        rolling Sharpe, return distribution.

plot_weights_and_risk   2-panel: dynamic allocation over time +
                        % risk contribution (Ledoit-Wolf covariance).
"""

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf


def plot_performance(
    ph: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    title: str = "Portfolio Performance",
) -> None:
    """
    4-panel performance dashboard.

    Panels
    ------
    1. Cumulative return vs. SPY benchmark
    2. Drawdown % (filled area)
    3. Rolling 63-day Sharpe ratio with threshold lines
    4. Return distribution with Jarque-Bera normality test

    Parameters
    ----------
    ph : pd.DataFrame
        Portfolio history from PortfolioBacktester.run(), must contain
        a 'portfolio_value' column.
    benchmark : pd.Series, optional
        Daily benchmark returns (e.g. SPY price changes).
    title : str
        Figure title.
    """
    ret = ph["portfolio_value"].pct_change().dropna()
    rf = 0.04 / 252

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 1. Cumulative return
    ax = axes[0, 0]
    norm = (1 + ret).cumprod()
    ax.plot(norm.index, norm, label="Strategy", color="#2E86AB", linewidth=2)
    if benchmark is not None:
        b = benchmark.reindex(ret.index).dropna()
        b_norm = (1 + b).cumprod() / (1 + b).cumprod().iloc[0]
        ax.plot(b_norm.index, b_norm, label="SPY", color="#A23B72",
                linestyle="--", linewidth=2)
    ax.set_title("Cumulative Return")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}×"))

    # 2. Drawdown
    ax = axes[0, 1]
    cum = (1 + ret).cumprod()
    dd = (cum - cum.expanding().max()) / cum.expanding().max() * 100
    ax.fill_between(dd.index, dd, 0, alpha=0.35, color="red")
    ax.plot(dd.index, dd, color="darkred", linewidth=1.2)
    ax.set_title("Drawdown (%)")
    ax.grid(alpha=0.3)

    # 3. Rolling 63-day Sharpe
    ax = axes[1, 0]
    excess = ret - rf
    rs = (
        excess.rolling(63).mean() * 252
        / (excess.rolling(63).std() * np.sqrt(252))
    )
    ax.plot(rs.index, rs, color="#006E90", linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.6)
    ax.axhline(1, color="green", linestyle=":", alpha=0.5, label="SR = 1")
    ax.set_title("Rolling 63d Sharpe")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Return distribution
    ax = axes[1, 1]
    ret_pct = ret * 100
    ax.hist(ret_pct, bins=60, alpha=0.7, color="#4A90E2", edgecolor="white")
    ax.axvline(ret_pct.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {ret_pct.mean():.3f}%")
    _, jb_p = stats.jarque_bera(ret_pct)
    ax.text(
        0.03, 0.96,
        f"Skew: {ret_pct.skew():.3f}\nKurt: {ret_pct.kurt():.3f}\nJB p: {jb_p:.4f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_title("Return Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_weights_and_risk(
    weights_history: pd.DataFrame,
    returns: pd.DataFrame,
    last_weights: pd.Series,
) -> None:
    """
    2-panel: portfolio allocation over time and % risk contribution.

    Panel 1 — Stacked area chart of dynamic asset weights (only assets
              with mean weight > 0.5% are shown to avoid clutter).

    Panel 2 — Horizontal bar chart of each asset's % contribution to
              total portfolio variance, using Ledoit-Wolf covariance.

    Parameters
    ----------
    weights_history : pd.DataFrame
        Date-indexed weight snapshots.
    returns : pd.DataFrame
        Daily returns used to estimate the covariance matrix.
    last_weights : pd.Series
        Most recent rebalance weights (for risk decomposition).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Weight evolution
    active = weights_history.loc[:, weights_history.mean() > 0.005]
    ax1.stackplot(
        active.index, [active[c] for c in active.columns],
        labels=active.columns, alpha=0.8,
    )
    ax1.set_title("Dynamic Portfolio Allocation", fontweight="bold")
    ax1.set_ylabel("Weight")
    ax1.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    ax1.grid(alpha=0.3)

    # Risk contribution
    common = returns.columns.intersection(last_weights.index)
    w = last_weights.reindex(common).fillna(0)
    w /= w.sum()
    cov = pd.DataFrame(
        LedoitWolf().fit(returns[common].dropna()).covariance_ * 252,
        index=common, columns=common,
    )
    port_var = float(w.values @ cov.values @ w.values)
    pct_rc = pd.Series(w.values * (cov.values @ w.values) / port_var, index=common)

    pct_rc.sort_values().plot(kind="barh", ax=ax2, color="#2E86AB")
    ax2.set_title(
        f"% Risk Contribution  (Port. Vol: {np.sqrt(port_var):.2%})",
        fontweight="bold",
    )
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
