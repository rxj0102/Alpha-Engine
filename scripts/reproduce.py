"""Run the walk-forward backtest with the notebook configuration and write results/.

Writes:
  results/metrics.json      strategy, BL-only ablation and SPY statistics
  results/ic_summary.csv    per-asset cross-fitted OOF IC across rebalances
  results/daily_returns.csv strategy and SPY daily returns

The README Results block is rendered from these files by scripts/render_readme.py.

Data: yfinance adjusted close (free, no key). Yahoo can serve slightly different
history on different days, which moved per-asset IC between two runs, and the
ETF market caps used as Black-Litterman prior weights are live values. Both
inputs are frozen in data/snapshot/ and reused. Pass --refresh to download
again and overwrite the snapshot.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.backtest import StatisticalTests  # noqa: E402
from src.config import BacktestConfig, Config, MLConfig, OptimizerConfig  # noqa: E402
from src.data import DataEngine  # noqa: E402
from src.pipeline import run_walk_forward  # noqa: E402

# Identical to the configuration cell of notebooks/alpha_engine.ipynb
CFG = Config(
    backtest=BacktestConfig(
        start_date="2020-01-01", end_date="2024-12-31", initial_capital=1_000_000,
        train_window=504, rebalance_freq="ME", run_ablation=True,
    ),
    ml=MLConfig(horizon=21, n_cv_splits=5),
    optimizer=OptimizerConfig(
        risk_aversion=2.5, risk_free_rate=0.04, max_weight=0.30, view_confidence=0.65,
    ),
)


def _stats(ret: pd.Series, rf: float) -> dict:
    """Same formulas as PortfolioBacktester.metrics, applied to a return series."""
    rf_d = rf / 252
    ex = ret - rf_d
    cum = (1 + ret).cumprod()
    down = ex[ex < 0]
    return {
        "Ann. Return": float(ret.mean() * 252),
        "Ann. Volatility": float(ret.std() * np.sqrt(252)),
        "Sharpe Ratio": float(ex.mean() * 252 / (ex.std() * np.sqrt(252))),
        "Sortino Ratio": float(ex.mean() * 252 / (down.std() * np.sqrt(252))),
        "Max Drawdown": float(((cum - cum.cummax()) / cum.cummax()).min()),
    }


def _parse(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        s = str(v)
        out[k] = float(s.rstrip("%")) / 100 if s.endswith("%") else float(s)
    return out


SNAP = ROOT / "data" / "snapshot"


def use_snapshot(refresh: bool) -> None:
    """Route DataEngine downloads through the committed snapshot files."""
    get_prices, get_caps = DataEngine.fetch_universe, DataEngine.fetch_market_caps
    SNAP.mkdir(parents=True, exist_ok=True)
    pf, cf = SNAP / "prices.csv", SNAP / "market_caps.csv"

    def fetch_universe(self, tickers):
        if refresh or not pf.exists():
            get_prices(self, tickers).to_csv(pf, float_format="%.6f", index_label="Date")
        df = pd.read_csv(pf, index_col="Date", parse_dates=True)
        df = df[[t for t in tickers if t in df.columns]]
        self.data["prices"] = df
        return df

    def fetch_market_caps(self, tickers):
        if refresh or not cf.exists():
            get_caps(self, tickers).rename("market_cap").to_csv(cf, index_label="ticker")
        return pd.read_csv(cf, index_col="ticker")["market_cap"].reindex(tickers)

    DataEngine.fetch_universe = fetch_universe
    DataEngine.fetch_market_caps = fetch_market_caps


def rounded(x):
    """Round floats to 8 significant digits so reruns write byte-identical files."""
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, float):
        return float(f"{x:.8g}")
    return x


def main() -> None:
    use_snapshot(refresh="--refresh" in sys.argv)
    res = run_walk_forward(CFG)
    rf = CFG.optimizer.risk_free_rate
    port = res["port_rets"]
    spy = res["spy_returns"].reindex(port.index).dropna()

    strat = _parse(res["metrics"])
    abl = _parse(res["ablation_metrics"])
    bench = _stats(spy, rf)
    bench.update({"Beta": 1.0})

    bbs = StatisticalTests.block_bootstrap_sharpe(port, rf=rf)
    n_trials = len(CFG.universe)
    dsr = StatisticalTests.deflated_sharpe_ratio(
        bbs["observed_sr"], n_trials, len(port),
        skewness=float(port.skew()), excess_kurtosis=float(port.kurt()),
    )

    rows = []
    for t, ics in res["ic_history"].items():
        a = np.asarray(ics, dtype=float)
        if len(a):
            rows.append({"ticker": t, "n_rebalances": len(a), "mean_IC": a.mean(),
                         "share_IC_positive": (a > 0).mean()})
    ic = pd.DataFrame(rows)
    all_ic = np.concatenate([np.asarray(v, float) for v in res["ic_history"].values() if len(v)])

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    ic.round(4).to_csv(out / "ic_summary.csv", index=False)
    pd.DataFrame({"strategy": port, "spy": spy}).to_csv(
        out / "daily_returns.csv", float_format="%.8f", index_label="date")
    payload = {
        "config": {"start": CFG.backtest.start_date, "end": CFG.backtest.end_date,
                   "train_window": CFG.backtest.train_window, "horizon": CFG.ml.horizon,
                   "rebalance": CFG.backtest.rebalance_freq, "risk_free": rf},
        "sample": {"first_day": str(port.index[0].date()), "last_day": str(port.index[-1].date()),
                   "n_days": int(len(port))},
        "strategy": strat,
        "ablation_bl_only": abl,
        "spy": bench,
        "tests": {"bootstrap_sharpe": float(bbs["observed_sr"]),
                  "bootstrap_p": float(bbs["p_value"]),
                  "dsr": float(dsr["DSR"]), "dsr_threshold": float(dsr["sr_threshold_ann"]),
                  "dsr_n_trials": n_trials},
        "ic": {"n_asset_rebalances": int(len(all_ic)), "mean": float(all_ic.mean()),
               "median": float(np.median(all_ic)), "share_positive": float((all_ic > 0).mean())},
    }
    payload = rounded(payload)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
