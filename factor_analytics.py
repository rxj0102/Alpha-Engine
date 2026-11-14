"""
Factor Analytics Module — Alpha Engine
=======================================
Computes IC/ICIR, factor decay, Fama-MacBeth regression, factor correlation,
quintile portfolios, Deflated Sharpe Ratio, and turnover analysis.

All charts saved to outputs/factor_analytics/ as 150-dpi PNGs.
Results table written to README_ANALYTICS.md.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("outputs/factor_analytics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "XLF", "XLE", "XLK", "XLV", "XLI",
    "TLT", "IEF", "LQD", "GLD", "USO",
]
ROLLING_WINDOW = 252
LAGS = [1, 5, 10, 20]
N_QUINTILES = 5

sns.set_theme(style="whitegrid", palette="tab10")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_prices(tickers: list[str], start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download adjusted close prices via yfinance; try local src.data first."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.data import DataEngine
        eng = DataEngine(tickers)
        prices = eng.fetch_universe(start, end)
        print(f"Loaded {prices.shape} price matrix from DataEngine.")
        return prices
    except Exception:
        pass

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices = prices.ffill().dropna(how="all")
    print(f"Loaded {prices.shape} price matrix from yfinance.")
    return prices


# ---------------------------------------------------------------------------
# 2. Feature engineering (18-factor set, mirrors src/data.py)
# ---------------------------------------------------------------------------

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build cross-sectional factor matrix: rows = dates, cols = (ticker, factor).
    Returns a MultiIndex-column DataFrame.
    """
    frames = []
    for ticker in prices.columns:
        p = prices[ticker].dropna()
        df = pd.DataFrame(index=p.index)

        # Momentum
        for w in [5, 21, 63, 252]:
            df[f"mom_{w}d"] = p.pct_change(w)

        # Realised volatility (annualised)
        for w in [5, 21, 63, 252]:
            df[f"rvol_{w}d"] = p.pct_change().rolling(w).std() * np.sqrt(252)

        # Trend
        sma20 = p.rolling(20).mean()
        sma50 = p.rolling(50).mean()
        df["price_to_sma20"] = p / sma20 - 1
        df["sma20_to_sma50"] = sma20 / sma50 - 1
        ema12 = p.ewm(span=12, adjust=False).mean()
        ema26 = p.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        df["macd"] = macd
        df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

        # Bollinger width
        bb_std = p.rolling(20).std()
        df["boll_width"] = 2 * bb_std / sma20

        # RSI (Wilder)
        delta = p.diff()
        up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        dn = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
        df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

        # Rolling stats
        ret = p.pct_change()
        df["roll63_mean"] = ret.rolling(63).mean()
        df["roll63_skew"] = ret.rolling(63).skew()
        df["roll63_kurt"] = ret.rolling(63).kurt()

        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        frames.append(df)

    result = pd.concat(frames, axis=1).sort_index()
    return result


def build_forward_returns(prices: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Cross-sectional forward log-returns shifted by `lag` trading days."""
    log_ret = np.log(prices / prices.shift(1))
    fwd = log_ret.shift(-lag)
    return fwd


# ---------------------------------------------------------------------------
# 3. Rolling Information Coefficient (IC)
# ---------------------------------------------------------------------------

def rolling_ic(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int = 1,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """
    For each signal compute rolling Spearman IC between signal(t) and
    forward_return(t + lag).  Returns DataFrame[date, signal].
    """
    fwd = build_forward_returns(prices, lag)
    tickers = prices.columns.tolist()
    factor_names = features.columns.get_level_values(1).unique().tolist()

    ic_rows = {}
    dates = features.index

    for fname in factor_names:
        sig_mat = features.xs(fname, axis=1, level=1, drop_level=True).reindex(columns=tickers)
        fwd_mat = fwd.reindex(columns=tickers)

        ic_series = pd.Series(index=dates, dtype=float)
        for t_idx in range(window, len(dates)):
            win_sig = sig_mat.iloc[t_idx - window : t_idx]
            win_fwd = fwd_mat.iloc[t_idx - window : t_idx]

            sig_flat = win_sig.values.flatten()
            fwd_flat = win_fwd.values.flatten()
            mask = np.isfinite(sig_flat) & np.isfinite(fwd_flat)
            if mask.sum() < 10:
                continue
            rho, _ = stats.spearmanr(sig_flat[mask], fwd_flat[mask])
            ic_series.iloc[t_idx] = rho

        ic_rows[fname] = ic_series

    return pd.DataFrame(ic_rows)


def compute_icir(ic_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """ICIR = rolling mean(IC) / rolling std(IC)."""
    roll_mean = ic_df.rolling(window).mean()
    roll_std = ic_df.rolling(window).std()
    return roll_mean / roll_std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# 4. Factor decay analysis
# ---------------------------------------------------------------------------

def factor_decay(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    """
    Mean IC (full-sample Spearman) at each lag for every signal.
    Returns DataFrame[lag, signal].
    """
    tickers = prices.columns.tolist()
    factor_names = features.columns.get_level_values(1).unique().tolist()
    rows = []
    for lag in lags:
        fwd = build_forward_returns(prices, lag)
        row = {"lag": lag}
        for fname in factor_names:
            sig_mat = features.xs(fname, axis=1, level=1, drop_level=True).reindex(columns=tickers)
            sig_flat = sig_mat.values.flatten()
            fwd_flat = fwd.reindex(columns=tickers).values.flatten()
            mask = np.isfinite(sig_flat) & np.isfinite(fwd_flat)
            if mask.sum() < 10:
                row[fname] = np.nan
                continue
            rho, _ = stats.spearmanr(sig_flat[mask], fwd_flat[mask])
            row[fname] = rho
        rows.append(row)
    return pd.DataFrame(rows).set_index("lag")


# ---------------------------------------------------------------------------
# 5. Fama-MacBeth cross-sectional regression
# ---------------------------------------------------------------------------

def newey_west_se(x: np.ndarray, n_lags: int = 4) -> float:
    """Newey-West heteroskedasticity and autocorrelation consistent std error."""
    n = len(x)
    xc = x - x.mean()
    var = np.dot(xc, xc) / n
    for lag in range(1, n_lags + 1):
        cov = np.dot(xc[lag:], xc[:-lag]) / n
        weight = 1 - lag / (n_lags + 1)
        var += 2 * weight * cov
    return float(np.sqrt(max(var, 0) / n))


def fama_macbeth(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    At each date t regress cross-sectional forward returns on lagged factors.
    Returns DataFrame of time-series coefficient estimates.
    """
    tickers = prices.columns.tolist()
    fwd = build_forward_returns(prices, lag).reindex(columns=tickers)
    factor_names = features.columns.get_level_values(1).unique().tolist()
    dates = features.index

    coef_rows = []
    for date in dates:
        # Build cross-section: one row per ticker
        y_row = fwd.loc[date] if date in fwd.index else pd.Series(dtype=float)
        try:
            X = np.array([
                features.loc[date].xs(fname, level=1).reindex(tickers).values
                for fname in factor_names
            ]).T  # shape (n_assets, n_factors)
            y = y_row.values  # shape (n_assets,)
        except Exception:
            continue

        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < len(factor_names) + 1:
            continue

        X_m, y_m = X[mask], y[mask]
        X_c = np.column_stack([np.ones(len(y_m)), X_m])
        try:
            coefs, _, _, _ = np.linalg.lstsq(X_c, y_m, rcond=None)
        except np.linalg.LinAlgError:
            continue

        row = {"date": date}
        for i, name in enumerate(["intercept"] + factor_names):
            row[name] = coefs[i]
        coef_rows.append(row)

    if not coef_rows:
        return pd.DataFrame()
    return pd.DataFrame(coef_rows).set_index("date")


def fama_macbeth_summary(coef_df: pd.DataFrame, n_lags: int = 4) -> pd.DataFrame:
    """Summarise FM coefficients: mean, t-stat, Newey-West SE."""
    rows = []
    for col in coef_df.columns:
        series = coef_df[col].dropna().values
        if len(series) < 5:
            continue
        mean = series.mean()
        nw_se = newey_west_se(series, n_lags=n_lags)
        t_stat = mean / nw_se if nw_se > 0 else np.nan
        rows.append({
            "factor": col,
            "mean_coef": mean,
            "nw_se": nw_se,
            "t_stat": t_stat,
        })
    return pd.DataFrame(rows).set_index("factor")


# ---------------------------------------------------------------------------
# 6. Factor correlation matrix
# ---------------------------------------------------------------------------

def factor_correlation(features: pd.DataFrame) -> pd.DataFrame:
    """Full-sample Spearman correlation across all signals (pooled cross-section)."""
    factor_names = features.columns.get_level_values(1).unique().tolist()
    tickers = features.columns.get_level_values(0).unique().tolist()

    # Stack into long form: one row per (date, ticker)
    long = {}
    for fname in factor_names:
        col = features.xs(fname, axis=1, level=1, drop_level=True).reindex(columns=tickers)
        long[fname] = col.values.flatten()

    df_long = pd.DataFrame(long).dropna()
    # Spearman correlation on ranks
    ranked = df_long.rank()
    return ranked.corr(method="pearson")  # equivalent to Spearman on ranks


# ---------------------------------------------------------------------------
# 7. Quintile portfolio analysis
# ---------------------------------------------------------------------------

def quintile_returns(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    lag: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    For each factor, rank stocks into N_QUINTILES buckets; compute mean forward
    return per quintile over time.  Returns dict[factor] -> DataFrame[date, quintile].
    """
    tickers = prices.columns.tolist()
    fwd = build_forward_returns(prices, lag).reindex(columns=tickers)
    factor_names = features.columns.get_level_values(1).unique().tolist()
    result = {}

    for fname in factor_names:
        sig_mat = features.xs(fname, axis=1, level=1, drop_level=True).reindex(columns=tickers)
        q_returns_rows = []
        for date in sig_mat.index:
            sig = sig_mat.loc[date]
            fwd_row = fwd.loc[date] if date in fwd.index else pd.Series(dtype=float)
            mask = sig.notna() & fwd_row.notna()
            if mask.sum() < N_QUINTILES:
                continue
            sig_m, fwd_m = sig[mask], fwd_row[mask]
            labels = pd.qcut(sig_m, N_QUINTILES, labels=False, duplicates="drop")
            row = {"date": date}
            for q in range(N_QUINTILES):
                row[f"Q{q+1}"] = fwd_m[labels == q].mean()
            q_returns_rows.append(row)
        if q_returns_rows:
            result[fname] = pd.DataFrame(q_returns_rows).set_index("date")

    return result


# ---------------------------------------------------------------------------
# 8. Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> dict:
    """Bailey & Lopez de Prado (2014) / Harvey & Liu (2015) DSR."""
    g = 0.5772  # Euler-Mascheroni constant
    if n_trials > 1:
        E_max = (
            (1 - g) * stats.norm.ppf(1 - 1 / n_trials)
            + g * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        E_max = 0.0

    adj = 1 - skewness * observed_sharpe + (excess_kurtosis - 1) / 4 * observed_sharpe ** 2
    adj = max(adj, 1e-10)
    thr_ann = E_max / np.sqrt(n_obs) * np.sqrt(adj) * np.sqrt(252)
    dsr = observed_sharpe / max(thr_ann, 1e-10)
    return {"DSR": dsr, "sr_threshold_ann": thr_ann, "passes_dsr": dsr > 1.0}


def compute_quintile_sharpes(
    quintile_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute annualised Sharpe for Q1 and Q5 of each factor."""
    rows = []
    for fname, df in quintile_dict.items():
        for q in ["Q1", "Q5"]:
            if q not in df.columns:
                continue
            r = df[q].dropna()
            if len(r) < 20:
                continue
            sr = float(np.sqrt(252) * r.mean() / r.std())
            rows.append({"factor": fname, "quintile": q, "sharpe": sr, "n_obs": len(r)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Turnover analysis
# ---------------------------------------------------------------------------

def compute_turnover(weights_history: pd.DataFrame) -> pd.Series:
    """
    Fraction of portfolio that changes between consecutive rebalances.
    weights_history: DataFrame[date, ticker] of portfolio weights.
    """
    diff = weights_history.diff().abs().sum(axis=1) / 2
    diff.name = "turnover"
    return diff.dropna()


# ---------------------------------------------------------------------------
# 10. Plotting
# ---------------------------------------------------------------------------

def _savefig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_rolling_ic(ic_df: pd.DataFrame, lag: int) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(22, 14), sharex=True)
    axes = axes.flatten()
    factors = ic_df.columns.tolist()
    for i, fname in enumerate(factors[:20]):
        ax = axes[i]
        ax.plot(ic_df.index, ic_df[fname], lw=0.8, color="steelblue")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(fname, fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(len(factors), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Rolling IC (lag={lag}d, window={ROLLING_WINDOW}d)", fontsize=13)
    plt.tight_layout()
    _savefig(f"rolling_ic_lag{lag}.png")


def plot_icir(icir_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in icir_df.columns:
        ax.plot(icir_df.index, icir_df[col], lw=0.7, alpha=0.6, label=col)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="ICIR=0.5 threshold")
    ax.axhline(-0.5, color="red", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Rolling ICIR (all factors)", fontsize=12)
    ax.set_ylabel("ICIR")
    ax.legend(fontsize=6, ncol=4, loc="upper left")
    plt.tight_layout()
    _savefig("icir_all_factors.png")


def plot_factor_decay(decay_df: pd.DataFrame) -> None:
    top_factors = decay_df.abs().max().nlargest(10).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    for fname in top_factors:
        ax.plot(decay_df.index, decay_df[fname], marker="o", ms=5, label=fname)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast lag (days)")
    ax.set_ylabel("Mean IC (Spearman)")
    ax.set_title("Factor Decay: IC at lags 1, 5, 10, 20 (top-10 by peak IC)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _savefig("factor_decay.png")


def plot_factor_correlation(corr_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df, mask=mask, cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, linewidths=0.3,
        annot=False, ax=ax,
    )
    ax.set_title("Factor Correlation Matrix (Spearman, pooled cross-section)", fontsize=12)
    plt.tight_layout()
    _savefig("factor_correlation.png")


def plot_quintile_spread(quintile_dict: dict[str, pd.DataFrame], top_n: int = 6) -> None:
    """Plot Q1 vs Q5 cumulative return for the top N factors by Q5-Q1 spread."""
    spreads = {}
    for fname, df in quintile_dict.items():
        if "Q1" in df.columns and "Q5" in df.columns:
            spread = (df["Q5"] - df["Q1"]).mean()
            spreads[fname] = spread
    top = sorted(spreads, key=lambda x: abs(spreads[x]), reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, fname in enumerate(top):
        ax = axes[i]
        df = quintile_dict[fname]
        cum_q1 = (1 + df["Q1"].fillna(0)).cumprod() - 1
        cum_q5 = (1 + df["Q5"].fillna(0)).cumprod() - 1
        ax.plot(cum_q1.index, cum_q1, label="Q1 (bottom)", color="crimson")
        ax.plot(cum_q5.index, cum_q5, label="Q5 (top)", color="steelblue")
        ax.fill_between(cum_q5.index, cum_q1, cum_q5, alpha=0.2, color="green")
        ax.set_title(fname, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_ylabel("Cumulative return")
        ax.tick_params(labelsize=7)
    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Quintile Spread: Q5 (top) vs Q1 (bottom)", fontsize=13)
    plt.tight_layout()
    _savefig("quintile_spread.png")


def plot_fm_coefficients(fm_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    top = summary["t_stat"].abs().nlargest(8).index.tolist()
    top = [f for f in top if f in fm_df.columns]
    if not top:
        return
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axes = axes.flatten()
    for i, fname in enumerate(top[:8]):
        ax = axes[i]
        ax.plot(fm_df.index, fm_df[fname], lw=0.8, color="darkorange")
        ax.axhline(fm_df[fname].mean(), color="navy", ls="--", lw=1)
        ax.axhline(0, color="black", lw=0.5)
        t = summary.loc[fname, "t_stat"] if fname in summary.index else np.nan
        ax.set_title(f"{fname}\nt={t:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Fama-MacBeth Time-Series Coefficients (top 8 by |t|)", fontsize=12)
    plt.tight_layout()
    _savefig("fama_macbeth_coefficients.png")


def plot_turnover(turnover: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(turnover.index, turnover.values, width=5, color="teal", alpha=0.7)
    ax.axhline(turnover.mean(), color="red", ls="--", lw=1.2, label=f"Mean={turnover.mean():.2%}")
    ax.set_ylabel("Turnover (fraction)")
    ax.set_title("Portfolio Turnover at Each Rebalance")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend()
    plt.tight_layout()
    _savefig("turnover.png")


def plot_dsr_summary(dsr_rows: list[dict]) -> None:
    if not dsr_rows:
        return
    df = pd.DataFrame(dsr_rows)
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 5))
    colors = ["steelblue" if p else "crimson" for p in df["passes_dsr"]]
    ax.bar(df["label"], df["DSR"], color=colors)
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="DSR=1 threshold")
    ax.set_title("Deflated Sharpe Ratio by Factor Quintile Strategy")
    ax.set_ylabel("DSR")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    _savefig("deflated_sharpe_ratio.png")


# ---------------------------------------------------------------------------
# 11. Summary statistics
# ---------------------------------------------------------------------------

def ic_summary_table(ic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ic_df.columns:
        s = ic_df[col].dropna()
        if len(s) < 5:
            continue
        mean = s.mean()
        std = s.std()
        icir = mean / std if std > 0 else np.nan
        t, p = stats.ttest_1samp(s, popmean=0.0)
        rows.append({
            "factor": col,
            "mean_IC": round(mean, 4),
            "std_IC": round(std, 4),
            "ICIR": round(icir, 3),
            "t_stat": round(t, 2),
            "p_value": round(p, 4),
            "significant": "Yes" if p < 0.05 else "No",
        })
    return pd.DataFrame(rows).set_index("factor").sort_values("ICIR", ascending=False)


# ---------------------------------------------------------------------------
# 12. README generation
# ---------------------------------------------------------------------------

def write_readme(
    ic_summary: pd.DataFrame,
    fm_summary: pd.DataFrame,
    dsr_rows: list[dict],
    decay_df: pd.DataFrame,
) -> None:
    lines = [
        "# Factor Analytics — Results Summary",
        "",
        "Generated by `factor_analytics.py`.",
        "",
        "## 1. IC and ICIR Summary (lag=1d, rolling 252-day window)",
        "",
        ic_summary.to_markdown(),
        "",
        "## 2. Fama-MacBeth Regression Summary (lag=1d, Newey-West SE, 4 lags)",
        "",
        fm_summary.round(5).to_markdown() if not fm_summary.empty else "_No results._",
        "",
        "## 3. Factor Decay (mean IC at lags 1, 5, 10, 20 days — top 10 factors)",
        "",
        decay_df.T.nlargest(10, decay_df.index[0]).T.round(4).to_markdown(),
        "",
        "## 4. Deflated Sharpe Ratio (Q5 quintile strategy per factor)",
        "",
    ]
    if dsr_rows:
        dsr_df = pd.DataFrame(dsr_rows)[["label", "DSR", "sr_threshold_ann", "passes_dsr"]]
        dsr_df = dsr_df.rename(columns={
            "label": "Factor",
            "DSR": "DSR",
            "sr_threshold_ann": "SR*",
            "passes_dsr": "Passes?"
        }).set_index("Factor")
        lines.append(dsr_df.round(3).to_markdown())
    else:
        lines.append("_No DSR results._")

    lines += [
        "",
        "## 5. Charts",
        "",
        "All charts saved to `outputs/factor_analytics/`:",
        "",
        "| Chart | Description |",
        "|-------|-------------|",
        "| `rolling_ic_lag*.png` | Rolling IC for each signal at each forecast lag |",
        "| `icir_all_factors.png` | Rolling ICIR for all signals |",
        "| `factor_decay.png` | IC vs forecast horizon (signal persistence) |",
        "| `factor_correlation.png` | Spearman correlation heatmap across all signals |",
        "| `quintile_spread.png` | Q5 vs Q1 cumulative return (top 6 factors) |",
        "| `fama_macbeth_coefficients.png` | FM coefficient time series (top 8 by |t|) |",
        "| `deflated_sharpe_ratio.png` | DSR per factor quintile strategy |",
        "| `turnover.png` | Rebalance turnover over time |",
    ]

    Path("README_ANALYTICS.md").write_text("\n".join(lines))
    print("  Written: README_ANALYTICS.md")


# ---------------------------------------------------------------------------
# 13. Main
# ---------------------------------------------------------------------------

def run_analytics(
    prices: pd.DataFrame | None = None,
    weights_history: pd.DataFrame | None = None,
) -> dict:
    """
    Run full factor analytics pipeline.

    Parameters
    ----------
    prices          : optional pre-loaded price DataFrame
    weights_history : optional DataFrame[date, ticker] of portfolio weights
    """
    print("=" * 60)
    print("Factor Analytics Pipeline")
    print("=" * 60)

    # ---- Data ----
    if prices is None:
        prices = load_prices(UNIVERSE)

    print("Building feature matrix …")
    features = build_features(prices)
    print(f"  Features shape: {features.shape}")

    # ---- Rolling IC at multiple lags ----
    ic_results: dict[int, pd.DataFrame] = {}
    for lag in LAGS:
        print(f"Computing rolling IC at lag={lag}d …")
        ic_df = rolling_ic(features, prices, lag=lag)
        ic_results[lag] = ic_df
        plot_rolling_ic(ic_df, lag=lag)

    # Primary IC at lag=1 for summaries
    ic_lag1 = ic_results[1]
    icir_df = compute_icir(ic_lag1)
    print("Plotting ICIR …")
    plot_icir(icir_df)

    # ---- Factor decay ----
    print("Computing factor decay …")
    decay_df = factor_decay(features, prices, lags=LAGS)
    plot_factor_decay(decay_df)

    # ---- Factor correlation ----
    print("Computing factor correlation matrix …")
    corr_df = factor_correlation(features)
    plot_factor_correlation(corr_df)

    # ---- Quintile portfolios ----
    print("Building quintile portfolios …")
    q_dict = quintile_returns(features, prices, lag=1)
    plot_quintile_spread(q_dict)

    # ---- Fama-MacBeth ----
    print("Running Fama-MacBeth regressions …")
    fm_coef_df = fama_macbeth(features, prices, lag=1)
    fm_summary = fama_macbeth_summary(fm_coef_df) if not fm_coef_df.empty else pd.DataFrame()
    if not fm_coef_df.empty:
        plot_fm_coefficients(fm_coef_df, fm_summary)

    # ---- DSR ----
    print("Computing Deflated Sharpe Ratios …")
    q_sharpes = compute_quintile_sharpes(q_dict)
    dsr_rows = []
    n_obs = len(prices)
    n_trials = len(q_dict)
    for _, row in q_sharpes.iterrows():
        if row["quintile"] != "Q5":
            continue
        q5 = q_dict[row["factor"]]["Q5"].dropna()
        sk = float(stats.skew(q5))
        ku = float(stats.kurtosis(q5))
        dsr_res = deflated_sharpe_ratio(
            row["sharpe"], n_trials=n_trials, n_obs=n_obs,
            skewness=sk, excess_kurtosis=ku,
        )
        dsr_rows.append({"label": row["factor"], **dsr_res})
    plot_dsr_summary(dsr_rows)

    # ---- Turnover ----
    if weights_history is not None and not weights_history.empty:
        print("Computing turnover …")
        turnover = compute_turnover(weights_history)
        plot_turnover(turnover)
    else:
        print("No weights_history provided — generating synthetic equal-weight baseline …")
        # Equal-weight rebalanced monthly
        monthly_dates = prices.resample("ME").last().index
        ew_weights = pd.DataFrame(
            1 / len(prices.columns),
            index=monthly_dates,
            columns=prices.columns,
        )
        # Add small random perturbation to simulate real turnover
        rng = np.random.default_rng(42)
        noise = rng.uniform(-0.02, 0.02, size=ew_weights.shape)
        noisy = (ew_weights + noise).clip(lower=0)
        noisy = noisy.div(noisy.sum(axis=1), axis=0)
        turnover = compute_turnover(noisy)
        plot_turnover(turnover)

    # ---- IC summary table ----
    print("Assembling IC summary …")
    ic_summary = ic_summary_table(ic_lag1)

    # ---- README ----
    print("Writing README_ANALYTICS.md …")
    write_readme(ic_summary, fm_summary, dsr_rows, decay_df)

    print("=" * 60)
    print("Done. All outputs in outputs/factor_analytics/")
    print("=" * 60)

    return {
        "ic": ic_results,
        "icir": icir_df,
        "decay": decay_df,
        "corr": corr_df,
        "quintile_returns": q_dict,
        "fm_coef": fm_coef_df,
        "fm_summary": fm_summary,
        "ic_summary": ic_summary,
        "dsr": dsr_rows,
    }


if __name__ == "__main__":
    results = run_analytics()
