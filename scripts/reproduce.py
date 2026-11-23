"""Run the experiment and write results/.

  results/metrics.json       everything the README reports
  results/daily_returns.csv  daily net returns of every strategy and the US market
  results/ic_monthly.csv     monthly rank IC of each signal
  results/weights/*.csv      target weights per rebalance for each strategy
  results/figures/*.png      cumulative excess return and rolling IC

Data are free open series (FRED, Ken French Data Library) frozen in
data/snapshot. Pass --refresh to download again and overwrite the snapshot.
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

from src import backtest as bt  # noqa: E402
from src.config import BONDS, Config  # noqa: E402
from src.pipeline import CLASSES, FORECAST_STRATEGIES, MARKET, STRATEGIES, run_experiment  # noqa: E402

PERIODS = [("1996-2006", "1996-01-01", "2006-12-31"), ("2007-2009", "2007-01-01", "2009-12-31"),
           ("2010-2019", "2010-01-01", "2019-12-31"), ("2020-2026", "2020-01-01", "2026-12-31")]
PAIRS = [(n, "1/N") for n in FORECAST_STRATEGIES] + [("ML", "Momentum"), ("ML", "Momentum + carry")]


def rounded(x):
    """Round floats to 6 significant digits so reruns write identical files."""
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, list):
        return [rounded(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.6g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def carry_timing_test(sc: dict, prices: pd.DataFrame, rf: pd.Series) -> dict:
    """Does a bond's carry z-score predict its next month excess return (in volatility units)?

    Pooled over the three bond series; the standard error is clustered by month
    because the three returns of one month are highly correlated.
    """
    import statsmodels.api as sm
    idx, reb = prices.index, sc["reb"]
    loc = idx.get_indexer(reb)
    rf_cum = (1 + rf.fillna(0)).cumprod()
    rows = []
    for k, d in enumerate(reb[:-1]):
        if d < sc["eval_start"]:
            continue
        rf_m = rf_cum.iloc[loc[k + 1]] / rf_cum.iloc[loc[k]] - 1
        for t in BONDS:
            r = prices[t].iloc[loc[k + 1]] / prices[t].iloc[loc[k]] - 1
            rows.append({"date": d, "z": sc["z_carry"].loc[d, t],
                         "y": (r - rf_m) / (sc["vol"].loc[sc["prev"][k], t] / np.sqrt(12))})
    df = pd.DataFrame(rows).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df["z"])).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["date"])[0]})
    return {"n_obs": int(len(df)), "n_months": int(df["date"].nunique()), "slope": float(fit.params["z"]),
            "t_stat": float(fit.tvalues["z"]), "p_value": float(fit.pvalues["z"]),
            "corr": float(df["z"].corr(df["y"]))}


def figures(excess: pd.DataFrame, ic: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    colors = {"1/N": "#8a8980", "Momentum": "#2a78d6", "Momentum + carry": "#eb6834", "ML": "#1baf7a"}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink,
                         "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface})

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    cum = (1 + excess[list(colors)]).cumprod()
    last = cum.iloc[-1].sort_values()
    ys = np.log(last.values)
    for i in range(1, len(ys)):            # keep end labels from overlapping
        ys[i] = max(ys[i], ys[i - 1] + 0.1)
    for n, c in colors.items():
        ax.plot(cum.index, cum[n], color=c, linewidth=1.3, label=n)
    for n, yl in zip(last.index, np.exp(ys)):
        ax.annotate(f"{n}", (cum.index[-1], yl), xytext=(6, 0), textcoords="offset points",
                    va="center", color=ink, fontsize=9)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(cum.index[0], cum.index[-1] + pd.Timedelta(days=1500))
    ax.set_title("Growth of 1 in excess of Treasury bills, after costs (log scale)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", ncol=4, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "cumulative_excess.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    roll = ic[["Momentum", "ML"]].rolling(36).mean()
    ax.axhline(0, color=muted, linewidth=0.8)
    for n in ("Momentum", "ML"):
        ax.plot(roll.index, roll[n], color=colors[n], linewidth=2, label=n)
        ax.annotate(n, (roll.index[-1], roll[n].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=ink, fontsize=9)
    ax.set_xlim(roll.index[36], roll.index[-1] + pd.Timedelta(days=700))
    ax.set_title("Rank IC against next month's return, 36 month rolling mean", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "rolling_ic.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = Config()
    refresh = "--refresh" in sys.argv
    res = run_experiment(cfg, refresh=refresh)
    if refresh:  # record exactly which downloaded files the new snapshot came from
        from src.opendata import write_manifest
        write_manifest(ROOT / "results" / "open_data_manifest.json")
    sc, rets, rf = res["scores"], res["returns"], res["rf"]
    prices = res["panel"]["prices"].ffill()
    excess = rets.sub(rf.reindex(rets.index).fillna(0), axis=0)
    names = STRATEGIES + [MARKET]

    summary = {n: bt.summarize(rets[n], rf, res["turnover"].get(n)) for n in names}
    for n in names:
        ci = bt.sharpe_ci(excess[n], cfg.n_bootstrap, cfg.block)
        summary[n].update({"sharpe_lo": ci["lo"], "sharpe_hi": ci["hi"]})
    for n in STRATEGIES:
        w = res["weights"][n]
        summary[n].update({f"avg_{c}": float(w[t].sum(axis=1).mean()) for c, t in CLASSES.items()})

    periods = {lab: {n: bt.summarize(rets[n].loc[a:b], rf)["sharpe"] for n in names} for lab, a, b in PERIODS}
    diffs = [{"a": a, "b": b, **bt.sharpe_diff_test(excess[a], excess[b], cfg.n_bootstrap, cfg.block)}
             for a, b in PAIRS]

    # Value added over the prior portfolio: monthly active return against 1/N.
    # A long-only portfolio earns the market premium whatever its views, so the
    # Sharpe ratio itself says nothing about the forecasts. The Deflated Sharpe
    # Ratio is therefore applied to the active return, with the four forecast
    # strategies as the trials.
    mret = pd.DataFrame({n: bt.monthly(rets[n]) for n in STRATEGIES})
    act = mret[FORECAST_STRATEGIES].sub(mret["1/N"], axis=0)
    ir_m = act.mean() / act.std()
    active = {}
    for n in FORECAST_STRATEGIES:
        kw = dict(sr=float(ir_m[n]), n_obs=len(act), skew=float(act[n].skew()), kurt=float(act[n].kurt() + 3))
        d = bt.deflated_sharpe(**kw, n_trials=len(FORECAST_STRATEGIES), trials_sr_std=float(ir_m.std()))
        active[n] = {"active_return_ann": float(act[n].mean() * 12), "tracking_error_ann": float(act[n].std() * np.sqrt(12)),
                     "info_ratio": float(ir_m[n] * np.sqrt(12)), "t_stat": float(ir_m[n] * np.sqrt(len(act))),
                     "psr": bt.deflated_sharpe(**kw)["dsr"], "dsr": d["dsr"]}
    active_meta = {"n_trials": len(FORECAST_STRATEGIES), "n_months": int(len(act)),
                   "sr0_annual": float(d["sr0"] * np.sqrt(12))}

    # rank IC of each score against next month's volatility-scaled return, out-of-sample dates only
    fc, fwd = sc["forecasts"], sc["fwd"]
    live = fc.index.get_level_values("date") >= sc["eval_start"]
    zm = sc["z_mom"].stack()
    zm.index.names = ["date", "ticker"]
    zm = zm[zm.index.get_level_values("date") >= sc["eval_start"]]
    ic = pd.DataFrame({"Momentum": bt.rank_ic(zm, fwd),
                       "ML ridge": bt.rank_ic(fc["ridge"][live], fwd),
                       "ML gradient boosting": bt.rank_ic(fc["gbm"][live], fwd),
                       "ML": bt.rank_ic(fc["ensemble"][live], fwd)})
    # Two checks on where the ML rank IC comes from. A score that only says
    # "prefer the low volatility assets" (bonds over oil) is a static tilt, not a
    # forecast; and within the nine equity series there is no asset class tilt to learn.
    vol_prev = sc["vol"].loc[sc["prev"]]
    vol_prev.index = sc["reb"]
    lowvol = (-vol_prev).stack()
    lowvol.index.names = ["date", "ticker"]
    lowvol = lowvol[lowvol.index.get_level_values("date") >= sc["eval_start"]]
    eq = lambda x: x[x.index.get_level_values("ticker").isin(CLASSES["equity"])]
    ic["Low volatility"] = bt.rank_ic(lowvol, fwd)
    ic["Momentum, equities only"] = bt.rank_ic(eq(zm), eq(fwd))
    ic["ML, equities only"] = bt.rank_ic(eq(fc["ensemble"][live]), eq(fwd))
    ic_stats = {c: bt.ic_stats(ic[c]) for c in ic.columns}
    ic_stats["ML minus Momentum"] = bt.ic_stats(ic["ML"] - ic["Momentum"])
    ic_stats["ML minus Low volatility"] = bt.ic_stats(ic["ML"] - ic["Low volatility"])
    ic_periods = {lab: {c: float(ic[c].loc[a:b].mean()) for c in ("Momentum", "ML")} for lab, a, b in PERIODS}
    ic_corr = float(ic["ML"].corr(ic["Momentum"]))
    score_corr = float(pd.concat([fc["ensemble"][live], zm], axis=1).dropna().corr(method="spearman").iloc[0, 1])

    robust = {lab: {n: bt.summarize(r, rf)["sharpe"] for n, r in d.items()} for lab, d in res["robust"].items()}

    # Partial trading (Garleanu and Pedersen 2013): each strategy trades a fixed
    # share of the way to its target every month. The difference against 1/N is
    # tested with the same paired bootstrap as the base case.
    rfa = rf.reindex(rets.index).fillna(0)
    partial = {}
    for k, runs in res["partial"].items():
        row = {}
        for n, r in runs.items():
            sm = bt.summarize(r["returns"], rf, r["turnover"])
            row[n] = {"sharpe": sm["sharpe"], "turnover_ann": sm["turnover_ann"]}
        for n in FORECAST_STRATEGIES:
            d = bt.sharpe_diff_test(runs[n]["returns"] - rfa.reindex(runs[n]["returns"].index).fillna(0),
                                    runs["1/N"]["returns"] - rfa.reindex(runs["1/N"]["returns"].index).fillna(0),
                                    cfg.n_bootstrap, cfg.block)
            row[n].update({"diff_vs_1n": d["diff"], "p_vs_1n": d["p_value"]})
        partial[str(k)] = row

    out = ROOT / "results"
    (out / "weights").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rets.to_csv(out / "daily_returns.csv", float_format="%.6f", index_label="date")
    ic.round(4).to_csv(out / "ic_monthly.csv", index_label="rebalance")
    slug = lambda n: n.lower().replace("/", "_over_").replace(" + ", "_").replace(" ", "_").replace("(", "").replace(")", "")
    for n in FORECAST_STRATEGIES:
        res["weights"][n].round(4).to_csv(out / "weights" / f"{slug(n)}.csv", index_label="rebalance")
    figures(excess, ic, out / "figures")

    n_reb = int((sc["reb"] >= sc["eval_start"]).sum())
    first_train = int((sc["X"].index.get_level_values("date") < sc["eval_start"]).sum())
    payload = rounded({
        "config": {"data_start": str(prices.index[0].date()), "data_end": str(prices.index[-1].date()),
                   "universe": cfg.universe, "cov_window": cfg.cov_window, "vol_window": cfg.vol_window,
                   "min_train_months": cfg.min_train_months, "risk_aversion": cfg.risk_aversion,
                   "max_weight": cfg.max_weight, "view_confidence": cfg.view_confidence,
                   "assumed_ic": cfg.assumed_ic, "n_bootstrap": cfg.n_bootstrap, "block": cfg.block},
        "sample": {"eval_start": str(sc["eval_start"].date()), "first_day": str(rets.index[0].date()),
                   "last_day": str(rets.index[-1].date()), "n_days": int(len(rets)), "n_rebalances": n_reb,
                   "years": float(len(rets) / 252), "panel_rows": int(len(sc["X"])),
                   "training_rows_first": int(sc["forecasts"].shape[0] and
                                              (sc["X"].index.get_level_values("date")
                                               < sc["forecasts"].index.get_level_values("date").min()).sum()),
                   "rows_before_eval": first_train},
        "summary": summary, "periods": periods, "sharpe_diffs": diffs,
        "active": active, "active_meta": active_meta,
        "ic": ic_stats, "ic_periods": ic_periods, "ic_corr_ml_momentum": ic_corr, "score_corr_ml_momentum": score_corr,
        "carry_timing": carry_timing_test(sc, prices, rf),
        "robustness": robust,
        "partial_trading": partial,
    })
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("active", "ic", "robustness")}, indent=1))
    print(pd.DataFrame(payload["summary"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
