# Alpha Engine

[![CI](https://github.com/rxj0102/Alpha-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/Alpha-Engine/actions/workflows/ci.yml)

**A walk-forward, multi-regime systematic portfolio that pipes ML return forecasts through a Bayesian Black-Litterman framework to construct risk-adjusted allocations across 15 multi-asset ETFs.**

---

## Research Contributions

This project addresses three specific, well-documented failure modes in applied quantitative research:

**1. Look-ahead bias in ML-based portfolio studies.**
The most common flaw in published ML-finance backtests is feature engineering performed on the full sample before the walk-forward split, silently leaking future information. Here, `DataEngine.engineer_features()` accepts an explicit `cutoff_date` and is re-called inside every rebalance iteration, ensuring that momentum, volatility, and technical features are computed only from data available at decision time. `[WF-1]`

**2. Retraining bias in stacked ensembles.**
A stacked meta-learner trained on the same data as its base models learns to correct in-sample base-model residuals, not genuine out-of-sample errors. The protocol has two stages: the meta-learner is evaluated walk-forward on out-of-fold (OOF) base predictions, so the reported IC comes from rows neither the base models nor the meta-learner were fit on, then both are refitted on the full training window before deployment. `[ML-1]`

**3. Mis-scaled Black-Litterman prior uncertainty.**
The standard practitioner implementation sets `τ = 1/n_assets` - a heuristic with no statistical justification. The theoretically correct choice (Idzorek 2005, He & Litterman 1999) is `τ = 1/T`, where `T` is the number of observations. For a 504-day training window and 15 assets, this correction shifts `τ` by a factor of ~34×, producing a substantially less dogmatic posterior and materially different allocations. `[BL-1]`

---

## Methodology

### Pipeline

```
Market Data (15 ETFs, 2020-2024)
         │
         ▼
  Feature Engineering          [WF-1] cutoff_date enforced per rebalance
  18 factors / asset:          momentum (5/21/63/252d), realised vol,
  MACD, Bollinger, Wilder RSI, SMA ratios, 63d mean/skew/kurt
         │
         ▼
  ML Ensemble                  [ML-1] Two-stage OOF + full refit
  RF + GBM + SVR + Ridge  →   Ridge meta-learner on OOF predictions
  Metric: Spearman IC          Views forwarded to BL only if IC > 0
         │
         ▼
  GJR-GARCH(1,1,1)-skewt      Conditional vol forecast (5-day horizon)
                               Per-asset regime: high_vol if vol > 75th pct
         │
         ▼
  Black-Litterman              [BL-1] τ = 1/T (not 1/N)
  Equilibrium: Π = δΣw_mkt    Ω proportional (He & Litterman)
  Posterior: μ_BL              Weighted combination of prior and ML views
         │
         ├── low-vol regime  → Mean-Variance QP   (CVXPY, λ = 2.5)
         └── high-vol regime → Min-CVaR LP         (α = 5%, tail-robust)
         │
         ▼
  Event-driven Backtest        Per-ticker bid-ask (3-10 bps) + 5 bps commission
  Monthly rebalance            Turnover computed on actual trading-day calendar
         │
         ▼
  Statistical Validation
  ├── Block-bootstrap Sharpe   B=5000, block=21d, circular (preserves autocorr.)
  ├── Deflated Sharpe Ratio    Multiple-testing penalty (Harvey & Liu 2015)
  └── IC t-test                Per-asset ICIR significance
```

### Design decisions

| Component | Choice | Why not the alternative |
|---|---|---|
| Feature cutoff | `cutoff_date` per rebalance | Full-sample features leak future vol/momentum |
| ML stacking | Cross-fitted meta-learner then full refit | Scoring the meta-learner on the rows it was fit on gave mean IC 0.125 on pure noise (12 of 12 runs positive); see tests |
| Covariance | Ledoit-Wolf shrinkage | Sample covariance is rank-deficient at T=504, N=15 |
| τ scaling | `1/T`, not `1/N` | `1/N` overstates prior certainty by a factor of T/N ≈ 34 |
| Vol model | GJR-GARCH (not GARCH) | Captures leverage effect; standard GARCH underestimates equity downside vol |
| High-vol objective | Min-CVaR LP | CVaR is a coherent risk measure; MV ignores tail shape under non-normality |
| Sharpe test | Block bootstrap | i.i.d. bootstrap under-rejects for autocorrelated monthly-rebalanced returns |
| Significance guard | Deflated SR | Raw Sharpe does not account for the number of strategies evaluated |

### Asset universe

| Class | Tickers | Role in portfolio |
|---|---|---|
| Broad equities | SPY, QQQ, IWM, EFA, EEM | Core equity beta; cross-sectional momentum signal |
| Sector ETFs | XLF, XLE, XLK, XLV, XLI | Sector rotation |
| Fixed income | TLT, IEF, LQD | Duration and credit risk-off allocation |
| Commodities | GLD, USO | Inflation hedge; low equity correlation in crisis |

---

## Mathematical Framework

### Black-Litterman posterior

The BL model treats return estimation as a Bayesian inference problem. The prior is the CAPM equilibrium implied by market-cap weights; the likelihood is formed from ML views. In practice Yahoo Finance publishes no market cap or share count for ETFs, so `fetch_market_caps` falls back to equal weights for every ticker in the default universe and the prior is the equal-weight equilibrium. The results below use that prior (see `data/snapshot/market_caps.csv`). The posterior is the minimum-variance blend:

$$\mu_{BL} = \bigl[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\bigr]^{-1} \bigl[(\tau\Sigma)^{-1}\Pi + P^\top\Omega^{-1}Q\bigr]$$

where:
- $\Pi = \delta\Sigma w_{mkt}$ - reverse-optimised equilibrium returns
- $\tau = 1/T$ - prior uncertainty scaling (correction `[BL-1]`)
- $P$ - view pick matrix (identity rows for absolute views)
- $Q$ - ML-predicted returns (filtered to IC > 0 assets only)
- $\Omega = \left(\frac{1}{c} - 1\right)\tau P\Sigma P^\top$ - He & Litterman proportional uncertainty

As $T \to \infty$, $\tau \to 0$ and $\mu_{BL} \to \Pi$ (prior dominates). As view confidence $c \to 1$, $\Omega \to 0$ and $\mu_{BL} \to Q$ (views dominate).

### GJR-GARCH leverage effect

Standard GARCH treats positive and negative shocks symmetrically. GJR-GARCH adds an asymmetry term:

$$\sigma^2_t = \omega + \underbrace{(\alpha + \gamma\,\mathbf{1}[\varepsilon_{t-1}<0])}_{\text{news impact}} \varepsilon^2_{t-1} + \beta\sigma^2_{t-1}$$

$\gamma > 0$ (empirically 0.05-0.15 for equity ETFs) means negative surprises amplify conditional variance more than equivalent positive surprises - the well-documented leverage effect (Black 1976, Christie 1982). Persistence $= \alpha + \tfrac{1}{2}\gamma + \beta$; values near 1 indicate slow mean-reversion (typical for equities). Long-run variance $= \omega / (1 - \text{persistence})$.

### Min-CVaR as a coherent risk measure

Unlike variance, CVaR (Expected Shortfall) is a coherent risk measure (Artzner et al. 1999): it is sub-additive, meaning diversification always reduces it. The LP formulation (Rockafellar & Uryasev 2000) avoids the instability of empirical quantile estimation:

$$\min_{w,\,\eta,\,z}\; \eta + \frac{1}{\alpha T}\sum_{t=1}^{T} z_t \quad \text{s.t.}\quad z_t \geq -r_t^\top w - \eta,\; z_t \geq 0,\; \mathbf{1}^\top w = 1,\; 0 \leq w_i \leq \bar{w}$$

where $\eta$ is the VaR and $z_t$ are auxiliary loss variables. This is used in high-vol regimes where tail shape matters more than second-moment approximations.

### Information Coefficient and the Fundamental Law

The Information Coefficient (IC) is the Spearman rank correlation between predicted and realised forward returns. Spearman is used instead of Pearson because it is robust to outliers and measures ordinal forecasting skill - what matters for cross-sectional ranking strategies. The ICIR = IC / σ(IC) is the per-asset analogue of the Sharpe ratio for the alpha signal.

Grinold's Fundamental Law of Active Management connects IC to strategy value:

$$IR \approx IC \cdot \sqrt{BR}$$

where BR is breadth (number of independent bets per year). Monthly rebalancing across 15 assets gives BR ≈ 180, meaning even a modest IC of 0.03 can support an IR of ~0.4 before transaction costs.

### Deflated Sharpe Ratio

The DSR (Harvey & Liu 2015) adjusts the significance threshold upward based on the number of configurations tested:

$$SR^* = \frac{\hat{E}[\max SR]}{\sqrt{T}} \cdot \sqrt{1 - \hat{\gamma}_3 \hat{SR} + \tfrac{\hat{\gamma}_4 - 1}{4}\hat{SR}^2}$$

where $\hat{E}[\max SR]$ is the expected maximum Sharpe across $N$ trials (approximated via the Euler-Mascheroni constant), and $\hat{\gamma}_3$, $\hat{\gamma}_4$ are return skewness and excess kurtosis. DSR > 1 means the observed Sharpe exceeds the threshold even after penalising for multiple testing.

---

## Results

Backtested 2020-01-01 to 2024-12-31 across the COVID crash, the 2022 rate shock and the 2023-24 recovery. The first rebalance is at the end of 2021, after the 504-day training window fills; before it the portfolio holds equal weights. Prices and the market-cap prior are frozen in `data/snapshot/` so the run is repeatable. `make reproduce` reruns everything (about 25 minutes on 2 cores) and rewrites the block below; CI checks that the README matches `results/metrics.json`.

<!-- results:start (generated by scripts/render_readme.py, do not edit by hand) -->
Sample: 2020-01-03 to 2024-12-30 (1256 trading days), monthly rebalance, 504-day training window, 21-day forecast horizon.

| Metric | ML + BL Strategy | BL-only (ablation) | SPY |
|---|---|---|---|
| Ann. Return | 8.89% | 9.85% | 15.65% |
| Ann. Volatility | 15.36% | 15.43% | 21.00% |
| Sharpe Ratio | 0.318 | 0.379 | 0.555 |
| Sortino Ratio | 0.382 | 0.455 | 0.680 |
| Max Drawdown | -28.90% | -28.90% | -33.72% |
| Beta | 0.639 | 0.653 | 1.000 |
| Alpha (ann.) | -1.12% | -0.37% | n/a |

- Effect of the ML views (ML + BL minus BL-only): Sharpe -0.061, annual return -0.96%.
- Block-bootstrap Sharpe test: SR 0.318, p = 0.277.
- Deflated Sharpe Ratio: 0.397 against a threshold of 0.803 (n_trials = 15).
- Cross-fitted out-of-fold IC over 540 asset-rebalance pairs: mean -0.023, median -0.030, positive in 45%. Only assets with positive IC at a rebalance send a view to Black-Litterman. Per-asset values are in results/ic_summary.csv.
<!-- results:end -->

How to read this:

- The out-of-fold IC of the stacked ensemble is centred near zero and below zero for most assets, so the ML forecasts have no detectable skill on 21-day ETF returns in this sample. Adding the ML views to Black-Litterman does not improve on the BL-only ablation.
- The Sharpe ratio is not significant under the block bootstrap and fails the deflated Sharpe test, so the strategy is not distinguishable from zero skill. It runs with lower volatility, drawdown and beta than SPY because of the position cap and the bond, gold and sector holdings, not because of the forecasts.
- The ablation is narrower than it looks: both arms optimise only over the assets whose IC passed the gate at that rebalance, so the BL-only column still uses the ML model to choose the asset set and isolates only the effect of the view returns.

## Corrections

Earlier versions of this README reported a Sharpe of 0.460 for ML + BL against 0.329 for BL-only, a bootstrap p-value of 0.000 and a mean IC of 0.44. Those came from the bugs below. The CV gap, the meta-learner scoring, the bootstrap and the feasibility rule have regression tests.

- CV leakage: `TimeSeriesSplit` now uses `gap = horizon` so that labels of the last training rows cannot overlap the test fold.
- In-sample meta-learner: the stacking Ridge was fit on the out-of-fold base predictions and then scored on the same rows, so it could flip the sign of any base model after seeing outcomes. The reported IC was positive even when all four base models had negative out-of-fold IC, and the pipeline gate (views only for assets with IC above zero) let all 15 assets through at all 36 rebalances in the pre-fix run. The meta-learner is now evaluated walk-forward (`_cross_fit_meta`). On pure-noise targets the old code gave mean IC 0.125 with 12 of 12 runs positive; the fixed code gives 0.003 with 6 of 12.
- Deployed meta-learner: it was fit on in-sample base predictions, which rewards the base model that overfits most. It is now fit on the out-of-fold base predictions, as in scikit-learn StackingRegressor.
- Block bootstrap: each resample was demeaned by its own mean, which made every null Sharpe exactly zero and returned p = 0 for any positive Sharpe. The series is now centred once before resampling.
- Infeasible optimisation: with exactly 3 assets passing the gate and a 30% cap, a fully invested portfolio is impossible, so the solver failed and the fallback held 33% in each asset. The pipeline now requires ceil(1 / max_weight) assets and otherwise holds the equal-weight universe.
- Selection bias in the printed IC test: the significance report tested only the assets that had passed the IC gate. It now uses every trained asset.

---

## Proposed Extensions

These are concrete next steps, not wishlist items. Each is tractable within the existing architecture.

### Extension 1 - Turnover-penalised mean-variance

The current MV objective ignores rebalancing costs. A quadratic turnover penalty transforms the problem into:

$$\max_w\; \mu^\top w - \frac{\lambda}{2} w^\top \Sigma w - \kappa \|w - w_{\text{prev}}\|_1$$

where $\kappa$ is a cost coefficient (set to ~10 bps × annualisation). This is still a convex QP solvable with CVXPY. Implementation: add `w_prev` as a parameter to `optimize_mv()` and extend the CVXPY constraint set.

### Extension 2 - Kalman filter view updating

Current ML views are generated monthly and held constant until the next rebalance. A state-space model allows continuous updating: let $\mu_t = \mu_{t-1} + w_t$ (random walk prior), and treat each day's cross-sectional prediction as a noisy observation $y_t = H\mu_t + v_t$. The Kalman update is:

$$\mu_{t|t} = \mu_{t|t-1} + K_t(y_t - H\mu_{t|t-1}), \quad K_t = P_{t|t-1}H^\top(HP_{t|t-1}H^\top + R)^{-1}$$

This replaces the discrete monthly view refresh with a continuous posterior update that decays stale signals exponentially. The noise ratio $Q/R$ controls how quickly views are updated versus held. This is the natural probabilistic extension of the BL framework and corresponds to treating the ML model as a measurement equation.

### Extension 3 - Statistical factor model covariance

Ledoit-Wolf works well for N ≤ 20, but a factor model is both more interpretable and more scalable. Decompose:

$$\Sigma = B F B^\top + D$$

where $B$ is the $N \times K$ factor loading matrix (estimated via PCA on returns), $F$ is the $K \times K$ factor covariance, and $D$ is a diagonal idiosyncratic covariance. With $K = 4$ factors (equity, duration, credit spread, commodity), this gives a covariance matrix with a natural economic interpretation: factor risk and idiosyncratic risk are explicitly separated. At monthly rebalancing, $K = 4$ factors capture >85% of cross-sectional variance in this universe. Implementation: replace `_cov()` in `optimizer.py` with a PCA factor decomposition, add a `FactorModel` class to `src/`, and add a `shrink_idio` flag to blend the factor model with Ledoit-Wolf for the residual component.

---

## Project Structure

```
Alpha-Engine/
├── src/
│   ├── __init__.py          # Public API: run_walk_forward, Config
│   ├── config.py            # BacktestConfig · MLConfig · OptimizerConfig
│   ├── data.py              # DataEngine: download + 18-factor engineering
│   ├── ml_alpha.py          # MLAlphaEngine: two-stage stacked ensemble
│   ├── volatility.py        # VolatilityEngine: GJR-GARCH + regime
│   ├── optimizer.py         # BlackLittermanOptimizer: BL · MV · CVaR · RP
│   ├── backtest.py          # PortfolioBacktester · StatisticalTests
│   ├── visualization.py     # 4-panel performance + risk decomposition
│   └── pipeline.py          # run_walk_forward orchestration
├── scripts/
│   ├── reproduce.py         # full backtest, writes results/
│   └── render_readme.py     # renders the Results block, checked in CI
├── results/                 # metrics.json, ic_summary.csv, daily_returns.csv
├── notebooks/
│   └── alpha_engine.ipynb   # Narrative walkthrough (imports from src/)
├── tests/
│   ├── test_data.py         # 11 tests: extraction, features, look-ahead
│   ├── test_ml_alpha.py     # 15 tests: IC, cross-fitted meta, train, predict
│   ├── test_optimizer.py    # 16 tests: equilibrium identity, bounds, ERC
│   └── test_backtest.py     # 22 tests: cost model, metrics, bootstrap, DSR
├── docs/
│   └── methodology.md       # Full technical methodology (research note style)
├── data/                    # Local cache (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/rxj0102/alpha-engine.git
cd alpha-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Run the pipeline:**
```python
from src import run_walk_forward
from src.config import Config, BacktestConfig, OptimizerConfig

results = run_walk_forward(Config(
    backtest=BacktestConfig(start_date="2020-01-01", end_date="2024-12-31"),
    optimizer=OptimizerConfig(risk_aversion=2.5, max_weight=0.30),
))
```

**Run tests:**
```bash
pytest tests/ -v
```

**Open the notebook:**
```bash
jupyter notebook notebooks/alpha_engine.ipynb
```

---

## Tech Stack

| Category | Library | Used for |
|---|---|---|
| Data | `yfinance` | Adjusted close (market cap retrieval returns nothing for ETFs) |
| ML | `scikit-learn` | RF, GBM, SVR, Ridge; StandardScaler; LedoitWolf |
| Volatility | `arch` | GJR-GARCH(1,1,1) with skewed-t distribution |
| Optimisation | `cvxpy` | MV QP and Min-CVaR LP (CLARABEL/OSQP/ECOS cascade) |
| Optimisation | `scipy.optimize` | Risk-parity SLSQP |
| Statistics | `scipy.stats` | Spearman IC, t-test, Jarque-Bera |
| Visualisation | `matplotlib`, `seaborn` | Performance dashboard, weight evolution |

---

## Reproducibility

### Data availability

All inputs come from free, public sources. No paid data or API keys are required unless noted.

- Yahoo Finance via the yfinance package (free, no key)
- `data/snapshot/prices.csv` and `data/snapshot/market_caps.csv`: the exact inputs behind the Results table, downloaded on 2026-11-10

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `uv.lock`.
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
pip install uv     # once
uv sync            # installs the exact pinned environment
uv run pytest -q   # runs the test suite
```

### Runtime

The test suite (64 tests) finishes in about 14 seconds on the machine above. `make reproduce`, the full walk-forward backtest from the frozen snapshot, takes about 23 minutes.

```bash
make reproduce   # rerun the backtest, write results/, refresh the README Results block
make check       # confirm README numbers match results/metrics.json (also run in CI)
uv run python scripts/reproduce.py --refresh   # download fresh prices into data/snapshot/ first
```

`notebooks/alpha_engine.ipynb` is the narrative walkthrough and imports the corrected code from `src/`. An older standalone notebook with pre-fix outputs was removed; it remains in git history.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## References

- Black, F. & Litterman, R. (1992). Global portfolio optimisation. *Financial Analysts Journal*, 48(5), 28-43.
- He, G. & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. *Goldman Sachs Investment Management Research*.
- Idzorek, T. (2005). A step-by-step guide to the Black-Litterman model. *Zephyr Associates Working Paper*.
- Glosten, L., Jagannathan, R. & Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779-1801.
- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21-41.
- Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13-28.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- Grinold, R. (1989). The fundamental law of active management. *Journal of Portfolio Management*, 15(3), 30-37.
- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203-228.