# Alpha Engine

**Systematic multi-asset portfolio construction combining ML return forecasts with Black-Litterman portfolio optimisation.**

---

## Motivation

Most quantitative strategies fall into one of two traps: pure ML approaches ignore portfolio construction entirely, while classical factor models lack the flexibility to incorporate modern predictive signals. This project bridges the gap — stacked ensemble forecasts are funnelled into a Bayesian Black-Litterman framework, producing a principled posterior over expected returns that respects market equilibrium while incorporating data-driven views.

The result is a fully walk-forward, multi-regime strategy with rigorous statistical validation that guards against the most common research pitfalls: look-ahead bias, overfitting, and p-hacking.

---

## Methodology

### Pipeline

```
Market Data (yfinance)
     │
     ▼
Feature Engineering  ──[WF-1: cutoff_date per rebalance]──►  18 factors / asset
     │
     ▼
ML Ensemble  ──[ML-1: two-stage OOF + full refit]──►  Return forecasts + IC
     │
     ▼
GJR-GARCH(1,1,1)  ──►  Conditional vol + regime (high / low)
     │
     ▼
Black-Litterman  ──[BL-1: τ = 1/T]──►  Posterior μ_BL
     │
     ├── low-vol regime  ──►  Mean-Variance QP  (CVXPY · CLARABEL/OSQP/ECOS)
     └── high-vol regime ──►  Min-CVaR LP       (tail-robust allocation)
     │
     ▼
Event-driven Backtest  ──►  Realistic costs (bid-ask + commission)
     │
     ▼
Statistical Validation  ──►  Block-bootstrap · Deflated SR · IC t-test
```

### Key Design Choices

| Component | Implementation | Rationale |
|---|---|---|
| Feature engineering | `cutoff_date` per rebalance [WF-1] | Eliminates look-ahead bias |
| ML ensemble | RF + GBM + SVR + Ridge → Ridge meta [ML-1] | OOF prevents retraining bias |
| Covariance | Ledoit-Wolf shrinkage (annualised) | Stable under T ≈ 500, N = 15 |
| BL prior scaling | τ = 1/T, not 1/N [BL-1] | Correct estimation uncertainty |
| Volatility model | GJR-GARCH(1,1,1) skewed-t | Leverage effect + fat tails |
| High-vol objective | Min-CVaR (LP) | More robust than MV in tail regimes |
| Sharpe test | Circular block bootstrap (block=21d) | Preserves return autocorrelation |
| Multiple-testing | Deflated Sharpe Ratio (Harvey & Liu 2015) | Guards against p-hacking |

### Asset Universe

| Class | Tickers |
|---|---|
| Broad Equities | SPY, QQQ, IWM, EFA, EEM |
| Sector ETFs | XLF, XLE, XLK, XLV, XLI |
| Fixed Income | TLT, IEF, LQD |
| Commodities | GLD, USO |

---

## Results

Backtested on **2020-01-01 – 2024-12-31** (includes COVID crash, 2022 rate shock, 2023–24 rally).

| Metric | ML + BL Strategy | BL-only (ablation) | SPY benchmark |
|---|---|---|---|
| Ann. Return | — | — | — |
| Ann. Volatility | — | — | — |
| Sharpe Ratio | — | — | — |
| Sortino Ratio | — | — | — |
| Max Drawdown | — | — | — |

> Results are populated when you run the pipeline. See `notebooks/alpha_engine.ipynb`.

**Statistical significance:**
- Block-bootstrap p-value (H₀: SR = 0) with B = 5,000 resamples
- Deflated Sharpe Ratio > 1.0 → survives multiple-testing penalty
- Per-asset IC t-test with ICIR reported

---

## Project Structure

```
Alpha-Engine/
├── src/
│   ├── __init__.py          # Public API
│   ├── config.py            # All parameters (BacktestConfig, MLConfig, OptimizerConfig)
│   ├── data.py              # DataEngine: price fetching + feature engineering
│   ├── ml_alpha.py          # MLAlphaEngine: stacked ensemble + IC
│   ├── volatility.py        # VolatilityEngine: GJR-GARCH + regime detection
│   ├── optimizer.py         # BlackLittermanOptimizer: BL + MV / CVaR / RP
│   ├── backtest.py          # PortfolioBacktester + StatisticalTests
│   ├── visualization.py     # Performance and risk decomposition charts
│   └── pipeline.py          # run_walk_forward: full orchestration
├── notebooks/
│   └── alpha_engine.ipynb   # Clean narrative walkthrough
├── tests/
│   ├── test_data.py         # DataEngine unit tests
│   ├── test_ml_alpha.py     # MLAlphaEngine unit tests
│   ├── test_optimizer.py    # BlackLittermanOptimizer unit tests
│   └── test_backtest.py     # Backtester + StatisticalTests unit tests
├── data/                    # Local cache (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/rxj0102/alpha-engine.git
cd alpha-engine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```python
from src import run_walk_forward
from src.config import Config, BacktestConfig

results = run_walk_forward(Config(
    backtest=BacktestConfig(start_date="2020-01-01", end_date="2024-12-31")
))
```

### 4. Or open the notebook

```bash
jupyter notebook notebooks/alpha_engine.ipynb
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Category | Library |
|---|---|
| Data | `yfinance` |
| ML | `scikit-learn` (RF, GBM, SVR, Ridge) |
| Volatility | `arch` (GJR-GARCH) |
| Optimisation | `cvxpy` (CLARABEL / OSQP / ECOS), `scipy` |
| Statistics | `scipy.stats` |
| Covariance | `sklearn.covariance.LedoitWolf` |
| Visualisation | `matplotlib`, `seaborn` |

---

## Mathematical Highlights

**Black-Litterman posterior (matrix form):**

$$\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\right]^{-1} \left[(\tau\Sigma)^{-1}\Pi + P^\top\Omega^{-1}Q\right]$$

where $\tau = 1/T$ (observations), $\Pi = \delta\Sigma w_{mkt}$, and $\Omega$ follows the He & Litterman proportional specification.

**GJR-GARCH leverage effect:**

$$\sigma^2_t = \omega + (\alpha + \gamma \cdot \mathbf{1}[\varepsilon_{t-1}<0])\varepsilon^2_{t-1} + \beta\sigma^2_{t-1}$$

$\gamma > 0$ means negative shocks amplify conditional variance more than positive shocks of equal magnitude.

**Deflated Sharpe Ratio** (Harvey & Liu 2015):

$$\widehat{SR}^* = \frac{E[\max SR]}{\sqrt{n_{obs}}} \cdot \sqrt{1 - \hat{\gamma}_3 \cdot \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4} \cdot \widehat{SR}^2}$$

The DSR threshold rises with the number of strategies tested, penalising data-mining.

---

## Future Improvements

- **Alternative data**: Incorporate NLP-derived earnings sentiment or options-implied moments as additional features
- **Factor covariance**: Replace Ledoit-Wolf with a Barra-style statistical factor model for larger universes
- **Dynamic views**: Kalman-filter-based online updating of BL views between rebalances
- **Turnover control**: Add L1 turnover penalty to the MV objective to reduce transaction costs
- **Market impact**: Almgren-Chriss cost model for large-notional portfolios
- **Hyperparameter search**: Optuna-based tuning of ensemble and GARCH parameters within the walk-forward loop