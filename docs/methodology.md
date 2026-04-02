# Technical Methodology

> This document describes the mathematical foundations, implementation decisions, and known limitations of the Alpha Engine. It is intended as a companion to the source code for readers who want to understand the "why" behind each design choice.

---

## 1. Problem Statement

The core challenge in systematic portfolio management is combining a noisy, short-horizon predictive signal (ML return forecast) with a long-horizon prior belief (CAPM equilibrium) in a way that is:

1. **Statistically principled** — the combination should weight each source by its precision.
2. **Economically coherent** — the resulting portfolio should reflect a defensible model of expected returns, not just a mechanical transformation of ML outputs.
3. **Empirically honest** — the evaluation must not inflate performance estimates through look-ahead bias, overfitting, or ignoring transaction costs.

The Black-Litterman framework addresses (1) and (2). The walk-forward protocol and statistical validation suite address (3).

---

## 2. Data and Feature Engineering

### 2.1 Universe and data quality

The universe consists of 15 liquid ETFs spanning equities, sectors, fixed income, and commodities. ETFs are chosen over individual stocks for three reasons: (i) they have daily liquidity far in excess of our assumed $1M AUM, making the frictionless-trading assumption more defensible; (ii) their bid-ask spreads are well-documented and stable; (iii) they provide clean exposure to the asset classes we want to model without idiosyncratic earnings risk.

Data is fetched per-ticker rather than in bulk to isolate individual download failures. Forward-filling with `limit=5` patches holiday-adjusted gaps without introducing stale prices for longer interruptions.

### 2.2 Feature construction

All 18 features are constructed from price and return data only. No macro variables or alternative data are used. Features fall into four groups:

**Momentum** (4 features): Return over 5, 21, 63, and 252 days. Short-horizon momentum (5d) captures mean-reversion signals in some assets; medium-horizon (21d) aligns with the holding period; long-horizon (252d) captures the cross-sectional momentum premium (Jegadeesh & Titman 1993).

**Realised volatility** (4 features): Annualised rolling standard deviation over the same four windows. Included as a risk-normalisation signal: an asset with high expected return but high vol may not offer better IC-adjusted forecasts than a lower-vol asset.

**Trend indicators** (4 features): Price-to-SMA20 and SMA20-to-SMA50 capture trend strength and crossover signals. MACD and its histogram measure the difference between short and long exponential moving averages, which is a standard measure of trend acceleration.

**Microstructure / distribution** (6 features): Bollinger band width measures realised vol relative to the price level. Wilder RSI (com=13 ≡ span=27) captures overbought/oversold conditions. The rolling 63-day mean, skewness, and excess kurtosis of returns capture the distribution of recent returns, which is informative about regime.

### 2.3 Look-ahead bias prevention [WF-1]

The single most common error in applied ML-finance is computing features over the entire sample and then performing a walk-forward split. This leaks future volatility, future momentum, and future distribution characteristics into the "past" training features.

The fix is simple: `engineer_features(cutoff_date=reb_date)` truncates the price series at the rebalance date before computing any rolling statistic. This means momentum, vol, RSI, and all other features at time $t$ are computed using only prices up to time $t$. The implementation resets `data_eng.data["prices"]` to the training window at each rebalance, then restores it — a careful but necessary pattern.

---

## 3. ML Alpha Generation

### 3.1 Why a stacked ensemble?

The return-prediction task is noisy (R² < 5% is typical), non-stationary, and multi-regime. No single model dominates across all conditions:

- Random Forest handles non-linear interactions and is robust to outliers, but can overfit to short training windows.
- Gradient Boosting corrects RF residuals sequentially and can capture regime-switching behaviour, but is slower and more sensitive to hyperparameters.
- SVR (RBF kernel) is effective in high-dimensional standardised feature spaces, and its large-margin formulation provides implicit regularisation.
- Ridge regression is the linear baseline. It is interpretable, low-variance, and often competitive with non-linear models in low-IC regimes.

The Ridge meta-learner learns the optimal time-varying mixture of these four signals from OOF predictions. Because the meta-learner has low capacity (linear), it is unlikely to overfit the mixing weights.

### 3.2 Two-stage training protocol [ML-1]

**The problem with naive stacking**: If we train the meta-learner on in-sample base predictions, the base models have already "seen" the labels and will produce artificially accurate predictions. The meta-learner then learns to weight models that appear good in-sample, which does not generalise.

**Stage 1 (evaluation)**:
1. Split the training data into 5 folds using `TimeSeriesSplit` (no shuffling — past to future ordering strictly maintained).
2. For each fold, fit base models on the training portion and predict on the held-out portion.
3. Assemble OOF predictions across all folds — every observation has exactly one prediction made using only past data.
4. Fit the meta-learner on these OOF predictions. The resulting IC and R² are unbiased estimates of out-of-sample performance.

**Stage 2 (deployment)**:
5. Refit all base models on the full training window.
6. Generate base predictions on the full training window (these are in-sample, which is intentional for Stage 2).
7. Refit the meta-learner on these full-data base predictions.

The deployed model is the Stage 2 model. It is trained on more data than any single fold in Stage 1, and its meta-learner is calibrated to the same base-model predictions it will see at inference time. Without Stage 2, we would deploy a model trained on ~80% of the training data (the last CV fold), creating a systematic discrepancy.

### 3.3 IC filter for view generation

At each rebalance, we compute OOF IC per asset. Only assets with IC > 0 have their ML-predicted return forwarded as a Black-Litterman view. This is a hard gate, not a soft weighting.

The rationale: a negative-IC asset's ML prediction is negatively correlated with future returns. Feeding it as a view would instruct the BL model to bet *against* the signal, which is correct in theory but requires confidence in the sign of the signal error — confidence we do not have. Gating on IC > 0 is conservative: it means we only express views when the OOF evidence supports positive predictability.

---

## 4. Volatility Modelling

### 4.1 GJR-GARCH specification

We fit a GJR-GARCH(1,1,1) model with skewed Student-t errors to each asset's daily return series:

$$\sigma^2_t = \omega + (\alpha + \gamma\,\mathbf{1}[\varepsilon_{t-1}<0])\varepsilon^2_{t-1} + \beta\sigma^2_{t-1}$$

The model is fit on returns scaled by ×100 to maintain numerical precision in the GARCH likelihood. All reported vol figures are back-scaled to decimal units and annualised by $\times\sqrt{252}$.

**Persistence** = $\alpha + \tfrac{1}{2}\gamma + \beta$. For typical equity ETFs, persistence ≈ 0.95–0.98, implying vol half-life of 14–34 trading days. Near-unit-root persistence (> 0.999) would indicate an IGARCH process where vol shocks are permanent — a model misspecification red flag.

**Long-run variance** = $\omega / (1 - \text{persistence})$. This is the unconditional variance to which conditional vol mean-reverts. It should roughly match the full-sample realised variance; large discrepancies suggest non-stationarity.

### 4.2 Regime detection

The binary high-vol / low-vol regime for each asset is derived by comparing the current 5-day forward vol forecast against the 75th percentile of the in-sample conditional vol history. Using the asset's own history makes the threshold adaptive: a 15% annualised vol forecast is "high" for TLT (historical median ~10%) but "low" for USO (historical median ~30%).

Portfolio regime is the majority vote across all assets with active ML views. Majority vote is more stable than asset-level regime switching and reduces false-positive high-vol signals from single-asset outliers.

---

## 5. Portfolio Construction

### 5.1 Ledoit-Wolf covariance shrinkage

The sample covariance is estimated from T ≈ 504 daily returns across N = 15 assets (T/N ≈ 34). This ratio is in the range where the sample covariance is estimable but noisy. Ledoit-Wolf (2004) provides the asymptotically optimal linear shrinkage toward a structured target (constant correlation) with analytically determined shrinkage intensity:

$$\hat{\Sigma}_{LW} = (1 - \alpha^*)\hat{S} + \alpha^* \mu_S I$$

where $\mu_S = \text{tr}(\hat{S})/N$ is the average variance and $\alpha^* \in [0,1]$ is the data-driven shrinkage coefficient. For T/N ≈ 34, we expect $\alpha^* \approx 0.05$–$0.15$. The estimator is used for both the BL equilibrium computation and the risk attribution in visualisation.

### 5.2 Black-Litterman posterior

The BL model is Bayes' theorem applied to expected return estimation:

$$\text{Prior: } \mu \sim \mathcal{N}(\Pi, \tau\Sigma), \quad \Pi = \delta\Sigma w_{mkt}$$
$$\text{Likelihood: } Q \sim \mathcal{N}(P\mu, \Omega)$$
$$\text{Posterior: } \mu_{BL} = M^{-1}\,\text{rhs}$$

where $M = (\tau\Sigma)^{-1} + P^\top\Omega^{-1}P$ is the posterior precision and $\text{rhs} = (\tau\Sigma)^{-1}\Pi + P^\top\Omega^{-1}Q$.

**Prior uncertainty**: $\tau = 1/T$ (see `[BL-1]` in `optimizer.py`). The standard practitioner setting $\tau = 1/N$ has no statistical basis. The correct interpretation of $\tau$ is the uncertainty in the prior mean: if $\Pi$ is estimated from a sample of size $T$, its standard error is $\Sigma/T$, so $\tau\Sigma = \Sigma/T$.

**View uncertainty**: $\Omega = (1/c - 1) \cdot \tau \cdot P\Sigma P^\top$. At $c = 0.5$, this gives $\Omega = \tau P\Sigma P^\top$, meaning prior and views have equal weight in the relevant direction. At $c = 0.65$, the views receive 1.86× as much weight as the prior in view space.

**Posterior intuition**: The posterior shrinks the ML view $Q_i$ toward the prior $\Pi_i$. The shrinkage factor for view $i$ depends on the ratio of view uncertainty ($\Omega_{ii}$) to prior uncertainty projected onto the view direction ($\tau [P\Sigma P^\top]_{ii}$). Views with high confidence (small $\Omega$) shift the posterior further from the prior.

### 5.3 Optimisation objectives

**Mean-variance (low-vol regime)**: The CAPM MV problem with a weight cap:

$$\max_w\; \mu_{BL}^\top w - \frac{\lambda}{2} w^\top\Sigma_{LW} w \quad \text{s.t.}\quad \mathbf{1}^\top w = 1,\; 0 \leq w_i \leq \bar{w}$$

The weight cap $\bar{w} = 0.30$ prevents degenerate corner solutions where MV concentrates into 1–2 assets with the highest IC-weighted return.

**Min-CVaR (high-vol regime)**: See Section 2 of `optimizer.py` for the LP derivation. Used when the portfolio regime is high-vol (majority of assets in the top volatility quartile). CVaR is preferred here because MV systematically underestimates portfolio risk when the return distribution has fat left tails — precisely the condition that triggers the high-vol regime.

---

## 6. Backtesting

### 6.1 Transaction cost model

Transaction costs are applied on every rebalance:

$$\text{cost}_i = |\Delta w_i| \cdot V \cdot \left(\frac{s_i}{2} + c\right)$$

where $\Delta w_i$ is the weight change for asset $i$, $V$ is portfolio value, $s_i$ is the per-ticker half bid-ask spread (3–10 bps), and $c = 5$ bps is the commission.

The half-spread model is appropriate for a market order-crossing strategy where each trade pays half the bid-ask on entry and half on exit. For limit orders or larger notionals, the effective cost would differ.

### 6.2 Rebalance calendar [BT-1]

Rebalance dates are derived from the actual price index by resampling with `pd.Series.resample("ME").last()`. This is critical: a synthetic `pd.date_range` would generate month-end dates that often fall on weekends or holidays, while the actual strategy would execute on the preceding trading day.

---

## 7. Statistical Validation

Three tests are run sequentially after the backtest. All three must be satisfied for the strategy to be considered robust.

### 7.1 Block-bootstrap Sharpe test

**H₀**: The true Sharpe Ratio is zero. The strategy's positive SR is due to chance.

The test generates 5,000 bootstrap null distributions by resampling contiguous blocks of 21 returns (with wrapping), centring each bootstrap sample at zero to enforce H₀. The p-value is the fraction of bootstrap SRs ≥ the observed SR.

The block size of 21 days (≈ 1 trading month) is chosen to preserve the autocorrelation structure of monthly-rebalanced strategies. A smaller block size would underestimate autocorrelation; a larger block size reduces the number of distinct bootstrap samples.

### 7.2 Deflated Sharpe Ratio

The DSR penalises the observed SR for the number of independent configurations evaluated. In this pipeline, `n_trials` = the number of assets with IC > 0. The interpretation: if 12 asset models are trained, we should expect the best-performing asset to have an inflated Sharpe by chance. The DSR threshold rises as $\ln(N)$, so it is conservative but not prohibitive.

DSR > 1 is a necessary but not sufficient condition for robustness. It should be read as: "the strategy's Sharpe is unlikely to be explained purely by data mining across assets."

### 7.3 IC significance

The mean IC across all rebalances is tested against zero using a one-sample t-test. The test is one-sided (positive IC is the hypothesis of interest). The t-statistic is $\bar{IC} / (\sigma_{IC} / \sqrt{n})$ where $n$ is the number of rebalances.

ICIR = $\bar{IC} / \sigma_{IC}$ is reported alongside the p-value. An ICIR > 0.5 with p < 0.05 provides strong evidence that the ML signal has genuine predictive content.

---

## 8. Limitations and Failure Modes

**Data source risk**: yfinance is not a professional data vendor. Adjusted close prices may differ across providers due to different dividend and split adjustment methodologies. Corporate actions (mergers, re-listings) can cause large gaps. For production use, a Bloomberg or Refinitiv feed is necessary.

**Single-country bias**: The universe is US-listed ETFs (including international equity ETFs, but traded in USD). This introduces implicit USD currency exposure and US market hours constraints. An extension to non-USD instruments would require FX hedging assumptions.

**Look-ahead in market caps**: Market caps are fetched at the time the script is run, not at each historical rebalance date. This is a minor source of look-ahead for the BL equilibrium prior, but its effect is small because market-cap weights change slowly.

**ML signal decay**: The 21-day forecast horizon is fixed. There is no adaptation to changing signal half-lives. In low-IC environments (e.g. early 2020 COVID period), views are likely noisy but still forwarded to BL as long as IC > 0 over the training window.

**Regime model simplicity**: The high-vol / low-vol binary regime based on a single vol threshold is crude. A hidden Markov model or multivariate regime-detection scheme (Hamilton 1989) would provide a richer, probabilistic regime classification.

**Transaction cost model**: Half bid-ask + commission is a simplified model. It ignores market impact (relevant at > $50M notional), intraday timing, and the option value of trading at the rebalance time rather than at close.

---

## 9. References

- Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.
- Black, F. (1976). Studies in stock price volatility changes. *Proceedings of the 1976 Business Meeting of the Business and Economic Statistics Section*, 177–181.
- Black, F., & Litterman, R. (1992). Global portfolio optimisation. *Financial Analysts Journal*, 48(5), 28–43.
- Christie, A. A. (1982). The stochastic behavior of common stock variances. *Journal of Financial Economics*, 10(4), 407–432.
- Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779–1801.
- Grinold, R. C. (1989). The fundamental law of active management. *Journal of Portfolio Management*, 15(3), 30–37.
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
- Harvey, C. R., & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1), 13–28.
- He, G., & Litterman, R. (1999). The intuition behind Black-Litterman model portfolios. *Goldman Sachs Investment Management Research*.
- Idzorek, T. (2005). A step-by-step guide to the Black-Litterman model. *Zephyr Associates Working Paper*.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Michaud, R. O. (1989). The Markowitz optimization enigma: Is optimized optimal? *Financial Analysts Journal*, 45(1), 31–42.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*, 2(3), 21–41.
