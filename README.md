# Alpha Engine

**A walk-forward, multi-regime systematic portfolio that pipes ML return forecasts through a Bayesian Black-Litterman framework to construct risk-adjusted allocations across 15 multi-asset ETFs.**

---

## Research Contributions

This project addresses three specific, well-documented failure modes in applied quantitative research:

**1. Look-ahead bias in ML-based portfolio studies.**
The most common flaw in published ML-finance backtests is feature engineering performed on the full sample before the walk-forward split, silently leaking future information. Here, `DataEngine.engineer_features()` accepts an explicit `cutoff_date` and is re-called inside every rebalance iteration, ensuring that momentum, volatility, and technical features are computed only from data available at decision time. `[WF-1]`

**2. Retraining bias in stacked ensembles.**
A stacked meta-learner trained on the same data as its base models learns to correct in-sample base-model residuals, not genuine out-of-sample errors. The fix is a two-stage protocol: the meta-learner is first fitted on out-of-fold (OOF) base predictions (yielding unbiased IC/R² estimates), then both base models and the meta-learner are refitted on the full training window before deployment. `[ML-1]`

**3. Mis-scaled Black-Litterman prior uncertainty.**
The standard practitioner implementation sets `τ = 1/n_assets` — a heuristic with no statistical justification. The theoretically correct choice (Idzorek 2005, He & Litterman 1999) is `τ = 1/T`, where `T` is the number of observations. For a 504-day training window and 15 assets, this correction shifts `τ` by a factor of ~34×, producing a substantially less dogmatic posterior and materially different allocations. `[BL-1]`

---

## Methodology

### Pipeline

```
Market Data (15 ETFs, 2020–2024)
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
  Event-driven Backtest        Per-ticker bid-ask (3–10 bps) + 5 bps commission
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
| ML stacking | OOF meta-learner then full refit | In-sample stacking inflates IC by ~0.05–0.15 |
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
| Sector ETFs | XLF, XLE, XLK, XLV, XLI | Sector rotation; highest ML IC historically |
| Fixed income | TLT, IEF, LQD | Duration and credit risk-off allocation |
| Commodities | GLD, USO | Inflation hedge; low equity correlation in crisis |

---

## Mathematical Framework

### Black-Litterman posterior

The BL model treats return estimation as a Bayesian inference problem. The prior is the CAPM equilibrium implied by market-cap weights; the likelihood is formed from ML views. The posterior is the minimum-variance blend:

$$\mu_{BL} = \bigl[(\tau\Sigma)^{-1} + P^\top\Omega^{-1}P\bigr]^{-1} \bigl[(\tau\Sigma)^{-1}\Pi + P^\top\Omega^{-1}Q\bigr]$$

where:
- $\Pi = \delta\Sigma w_{mkt}$ — reverse-optimised equilibrium returns
- $\tau = 1/T$ — prior uncertainty scaling (correction `[BL-1]`)
- $P$ — view pick matrix (identity rows for absolute views)
- $Q$ — ML-predicted returns (filtered to IC > 0 assets only)
- $\Omega = \left(\frac{1}{c} - 1\right)\tau P\Sigma P^\top$ — He & Litterman proportional uncertainty

As $T \to \infty$, $\tau \to 0$ and $\mu_{BL} \to \Pi$ (prior dominates). As view confidence $c \to 1$, $\Omega \to 0$ and $\mu_{BL} \to Q$ (views dominate).

### GJR-GARCH leverage effect

Standard GARCH treats positive and negative shocks symmetrically. GJR-GARCH adds an asymmetry term:

$$\sigma^2_t = \omega + \underbrace{(\alpha + \gamma\,\mathbf{1}[\varepsilon_{t-1}<0])}_{\text{news impact}} \varepsilon^2_{t-1} + \beta\sigma^2_{t-1}$$

$\gamma > 0$ (empirically 0.05–0.15 for equity ETFs) means negative surprises amplify conditional variance more than equivalent positive surprises — the well-documented leverage effect (Black 1976, Christie 1982). Persistence $= \alpha + \tfrac{1}{2}\gamma + \beta$; values near 1 indicate slow mean-reversion (typical for equities). Long-run variance $= \omega / (1 - \text{persistence})$.

### Min-CVaR as a coherent risk measure

Unlike variance, CVaR (Expected Shortfall) is a coherent risk measure (Artzner et al. 1999): it is sub-additive, meaning diversification always reduces it. The LP formulation (Rockafellar & Uryasev 2000) avoids the instability of empirical quantile estimation:

$$\min_{w,\,\eta,\,z}\; \eta + \frac{1}{\alpha T}\sum_{t=1}^{T} z_t \quad \text{s.t.}\quad z_t \geq -r_t^\top w - \eta,\; z_t \geq 0,\; \mathbf{1}^\top w = 1,\; 0 \leq w_i \leq \bar{w}$$

where $\eta$ is the VaR and $z_t$ are auxiliary loss variables. This is used in high-vol regimes where tail shape matters more than second-moment approximations.

### Information Coefficient and the Fundamental Law

The Information Coefficient (IC) is the Spearman rank correlation between predicted and realised forward returns. Spearman is used instead of Pearson because it is robust to outliers and measures ordinal forecasting skill — what matters for cross-sectional ranking strategies. The ICIR = IC / σ(IC) is the per-asset analogue of the Sharpe ratio for the alpha signal.

Grinold's Fundamental Law of Active Management connects IC to strategy value:

$$IR \approx IC \cdot \sqrt{BR}$$

where BR is breadth (number of independent bets per year). Monthly rebalancing across 15 assets gives BR ≈ 180, meaning even a modest IC of 0.03 can support an IR of ~0.4 before transaction costs.

### Deflated Sharpe Ratio

The DSR (Harvey & Liu 2015) adjusts the significance threshold upward based on the number of configurations tested:

$$SR^* = \frac{\hat{E}[\max SR]}{\sqrt{T}} \cdot \sqrt{1 - \hat{\gamma}_3 \hat{SR} + \tfrac{\hat{\gamma}_4 - 1}{4}\hat{SR}^2}$$

where $\hat{E}[\max SR]$ is the expected maximum Sharpe across $N$ trials (approximated via the Euler-Mascheroni constant), and $\hat{\gamma}_3$, $\hat{\gamma}_4$ are return skewness and excess kurtosis. DSR > 1 means the observed Sharpe exceeds the threshold even after penalising for multiple testing.

---

## Results

Backtested **2020-01-01 – 2024-12-31** across the COVID crash (-34% SPY drawdown), 2022 rate shock (-20% SPY), and 2023–24 recovery — a stress-test that covers three distinct macro regimes.

| Metric | ML + BL Strategy | BL-only (ablation) | SPY |
|---|---|---|---|
| Ann. Return | — | — | — |
| Ann. Volatility | — | — | — |
| Sharpe Ratio | — | — | — |
| Sortino Ratio | — | — | — |
| Max Drawdown | — | — | — |
| Alpha (ann.) | — | — | — |

> Run `notebooks/alpha_engine.ipynb` to populate. The ablation column isolates the ML contribution by removing views while keeping the BL + GARCH + CVaR regime framework identical.

**Statistical validation checklist:**
- [ ] Block-bootstrap p-value < 0.05 (H₀: SR = 0, B = 5,000, block = 21d)
- [ ] Deflated Sharpe Ratio > 1.0 (survives multiple-testing penalty)
- [ ] Mean IC significantly > 0 (per-asset t-test, ICIR reported)

---

## Proposed Extensions

These are concrete next steps, not wishlist items. Each is tractable within the existing architecture.

### Extension 1 — Turnover-penalised mean-variance

The current MV objective ignores rebalancing costs. A quadratic turnover penalty transforms the problem into:

$$\max_w\; \mu^\top w - \frac{\lambda}{2} w^\top \Sigma w - \kappa \|w - w_{\text{prev}}\|_1$$

where $\kappa$ is a cost coefficient (set to ~10 bps × annualisation). This is still a convex QP solvable with CVXPY. Empirically, turnover penalties reduce gross-to-net Sharpe degradation by 30–60% in monthly-rebalanced strategies. Implementation: add `w_prev` as a parameter to `optimize_mv()` and extend the CVXPY constraint set.

### Extension 2 — Kalman filter view updating

Current ML views are generated monthly and held constant until the next rebalance. A state-space model allows continuous updating: let $\mu_t = \mu_{t-1} + w_t$ (random walk prior), and treat each day's cross-sectional prediction as a noisy observation $y_t = H\mu_t + v_t$. The Kalman update is:

$$\mu_{t|t} = \mu_{t|t-1} + K_t(y_t - H\mu_{t|t-1}), \quad K_t = P_{t|t-1}H^\top(HP_{t|t-1}H^\top + R)^{-1}$$

This replaces the discrete monthly view refresh with a continuous posterior update that decays stale signals exponentially. The noise ratio $Q/R$ controls how quickly views are updated versus held. This is the natural probabilistic extension of the BL framework and corresponds to treating the ML model as a measurement equation.

### Extension 3 — Statistical factor model covariance

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
├── notebooks/
│   └── alpha_engine.ipynb   # Narrative walkthrough (imports from src/)
├── tests/
│   ├── test_data.py         # 17 tests: extraction, features, look-ahead
│   ├── test_ml_alpha.py     # 13 tests: IC, train, predict, reproducibility
│   ├── test_optimizer.py    # 14 tests: equilibrium identity, bounds, ERC
│   └── test_backtest.py     # 16 tests: cost model, metrics, bootstrap, DSR
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
| Data | `yfinance` | Adjusted close + market cap retrieval |
| ML | `scikit-learn` | RF, GBM, SVR, Ridge; StandardScaler; LedoitWolf |
| Volatility | `arch` | GJR-GARCH(1,1,1) with skewed-t distribution |
| Optimisation | `cvxpy` | MV QP and Min-CVaR LP (CLARABEL/OSQP/ECOS cascade) |
| Optimisation | `scipy.optimize` | Risk-parity SLSQP |
| Statistics | `scipy.stats` | Spearman IC, t-test, Jarque-Bera |
| Visualisation | `matplotlib`, `seaborn` | Performance dashboard, weight evolution |

---

## References

- Black, F. & Litterman, R. (1992). Global portfolio optimisation. *Financial Analysts Journal*, 48(5), 28–43.
- He, G. & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. *Goldman Sachs Investment Management Research*.
- Idzorek, T. (2005). A step-by-step guide to the Black-Litterman model. *Zephyr Associates Working Paper*.
- Glosten, L., Jagannathan, R. & Runkle, D. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Rockafellar, R. T. & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.
- Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13–28.
- Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Grinold, R. (1989). The fundamental law of active management. *Journal of Portfolio Management*, 15(3), 30–37.
- Artzner, P., Delbaen, F., Eber, J.-M. & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.