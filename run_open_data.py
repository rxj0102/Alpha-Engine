"""Run the walk-forward Alpha Engine on open data and save the metrics.

Usage: python run_open_data.py
Writes results/open_data_metrics.json and prints the README table.
"""
import json
import logging
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from src.backtest import StatisticalTests  # noqa: E402
from src.pipeline import run_walk_forward  # noqa: E402

np.random.seed(42)
out = run_walk_forward()
res = {"strategy": out["metrics"], "ablation": out.get("ablation_metrics"),
       "start": str(out["prices"].index[0].date()), "end": str(out["prices"].index[-1].date()),
       "assets": list(out["prices"].columns)}
spy = out["spy_returns"].dropna()
spy = spy.loc[out["port_rets"].index.min():out["port_rets"].index.max()]
res["spy"] = {"ann_return": float(spy.mean() * 252), "ann_vol": float(spy.std() * np.sqrt(252)),
              "sharpe": float((spy.mean() * 252 - 0.04) / (spy.std() * np.sqrt(252)))}
res["live_ic_stats"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in out["live_ic_stats"].items()}
res["live_ic"] = {str(k.date()): v for k, v in out["live_ic"].items()}
for k in ("stat_tests", "statistical_tests", "sharpe_test", "dsr"):
    if k in out:
        res[k] = out[k]
Path("results").mkdir(exist_ok=True)
Path("results/open_data_metrics.json").write_text(json.dumps(res, indent=2, default=float))
print(json.dumps(res, indent=2, default=float))
