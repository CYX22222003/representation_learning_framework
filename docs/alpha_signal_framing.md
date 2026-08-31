# Alpha-Research Downstream Capability

This note consolidates the planned alpha-research capability. It is a downstream capability test of the representation-learning framework, not the project's main methodological contribution and not a claim to have invented an alpha-mining algorithm or trading strategy.

## Core Idea

The framework should not be described as directly producing alpha factors from the raw feature branches. A more precise framing is:

```text
Raw OHLCV sequence X
  -> frozen multi-branch representation learning
  -> shared representation z
  -> economically meaningful downstream heads
  -> primitive prediction set F_0
  -> shallow symbolic alpha mining
  -> candidate formulaic factors {alpha_1, ..., alpha_K}
```

In this framing, the statistical, transformed, VAE, contrastive, and BYOL branches are reusable market representations. They encode information about price dynamics, volatility structure, frequency patterns, nonlinear temporal structure, and regime behavior. The downstream task head then converts those representations into task-specific outputs, such as predicted price movement, predicted volatility, or trend probability.

The latent coordinates `z_j` are deliberately **not** primitive alpha factors. Their axes are learned for representation utility, so they need not carry a stable or financially interpretable individual meaning. Directly mining arbitrary dimensions would make formula interpretation, stability analysis, and economic discussion substantially weaker.

Instead, the downstream task-head outputs form the primitive set \(\mathcal F_0\). They retain a clear predictive interpretation while allowing the representation to capture temporal structure flexibly. A prediction is still not an alpha factor by definition: a formula becomes a candidate alpha only after it demonstrates robust out-of-sample value for a predeclared objective.

## Terminology

| Term | Meaning in this project |
|---|---|
| Representation / feature | The extracted embedding or deterministic feature vector from each branch, before task supervision. |
| Downstream head | A lightweight supervised model trained on frozen representations for a specific task. |
| Predictive signal | The output of the downstream head, such as expected return, trend probability, or expected volatility. |
| Candidate alpha signal | A predictive signal that may be useful for alpha research after validation. |
| Alpha factor | A signal or factor that has demonstrated robust out-of-sample predictive value for a tradable objective. |
| Trading strategy | A complete rule set that converts signals into positions, sizing, risk control, and execution. |

This distinction is important because the project evaluates representation quality and signal usefulness. It does not yet claim to implement a complete trading strategy or production alpha pipeline.

## Primitive Factor Set

For each timestamp/contract row, construct \(\mathcal F_0\) only from predictions available at that row:

\[
\mathcal F_0 = \{\hat r_h,\ \widehat{\Delta p}_h,\ \hat\sigma_h,\ p_{\mathrm{bull},h},\ p_{\mathrm{bear},h},\ p_{\mathrm{neutral},h},\ p_{\mathrm{bull},h}-p_{\mathrm{bear},h},\ldots\},
\]

where \(h\) denotes a predeclared forecast horizon. The exact contents depend on which heads are implemented and whose targets are split-safe. The recommended initial set is deliberately small: a directional return/movement score, realised-volatility forecast, three trend probabilities, and the bullish-minus-bearish confidence score. Multiple horizons are a later extension, added only when their target definitions and availability times are fixed in advance.

### Decisions to Lock Before Implementation

- The directional head must have a single predeclared target: future probability change or a carefully defined return. A price-*level* forecast should not be used as a cross-contract factor without converting it to a decision-time comparable movement score.
- The factor objective, forecast horizon, rebalance/observation frequency, and eligible contract universe must agree. Contract expiry, missing observations, sparse trading, and overlapping windows require explicit handling.
- Cross-sectional ranking/IC is meaningful only for rows sharing a comparable timestamp and information set. If the available data cannot provide this, use a clearly stated within-contract time-series objective instead of presenting cross-sectional alpha evidence.

## Symbolic Alpha-Mining Scope

Use a compact genetic-programming (GP) or symbolic-regression search as an established, interpretable search tool. AutoAlpha (Zhang et al., 2020) supports the general idea that formulaic factors can be evolved from a predefined terminal and operator set; its hierarchical evolutionary search, large stock universe, portfolio construction, and novelty algorithm are **not** claims or required components of this project.

The initial grammar should contain protected arithmetic, cross-sectional ranking, delay, rolling mean/std, rolling rank, and a small set of predeclared time-series operators. Candidate examples are

\[
\hat r / \hat\sigma, \qquad
\hat r\,(p_{\mathrm{bull}}-p_{\mathrm{bear}}), \qquad
tsrank(p_{\mathrm{bull}}-p_{\mathrm{bear}}, 10).
\]

Cap tree depth and rolling windows in advance. Prefer a small population/generation budget and retain a small, non-redundant factor set rather than optimising a large expression library. Protected division and explicit handling of missing/near-zero volatility are required.

## Leakage-Safe Alpha-Research Protocol

GP selection is a model-selection procedure. It must never see locked test predictions, targets, ICs, or backtests.

1. Train each fixed downstream head on an earlier training prefix and emit predictions for a later training fold; repeat chronologically to build out-of-fold (OOF) predictions covering the usable training rows. Each prediction must come from a head that did not train on that row or any later row.
2. Fit GP only on these OOF primitive factors and their aligned training-period future-return objective. Standardise/rank using training-fold information only; apply purge/embargo around horizon overlap where needed.
3. Select a small factor set using a predeclared training criterion, such as mean Spearman IC plus stability across chronological folds, with an explicit correlation/redundancy constraint. Do not select by test performance.
4. For a future alpha study, refit the fixed downstream heads on the full training split, generate the chosen primitives and formulas on a fresh holdout or temporally later data, and evaluate the fixed formula set once. Do not reuse the current task-evaluation test split for factor selection or headline factor evidence.

This is cross-fitting for honest training predictions, analogous to the existing GARCH--LSTM meta-feature protocol. It is not an early-stopping or test-driven validation split.

## Alpha-Factor Evaluation

The primary framework metrics remain the supervised task metrics. The additional alpha-research capability adds factor-level evidence:

- cross-sectional Spearman IC and, where appropriate, Pearson IC against a predeclared future return/movement target;
- IC mean, dispersion, and stability across chronological folds or market regimes;
- monotonic quantile/spread analysis, using only cross-sectionally comparable contract rows;
- factor correlation and redundancy among retained formulas;
- only if contract mechanics, fees, liquidity, and position constraints are modelled: a clearly specified, cost-aware paper backtest.

Report the primitive-only signals alongside GP formulas. This distinguishes whether performance comes from the representation-generated predictions themselves or from symbolic composition. The alpha result is supportive downstream evidence; negative results do not invalidate the representation-learning contribution.

## How the Existing Tasks Map to Alpha Research

### Price Prediction

The price prediction head produces a forecast of future price level, probability level, or return depending on the chosen label. This can be interpreted as a candidate directional signal if it is transformed into:

- expected future return,
- expected price change,
- ranking score across contracts,
- long/short or buy/sell decision score.

This task is directly related to alpha research because it asks whether the learned representation contains information about future market movement.

### Trend Classification

The trend classification head produces a probability or class label for future direction, such as upward, downward, or neutral movement. This is the closest task to a traditional directional alpha signal because the output can be interpreted as a probability-weighted view of future direction.

Possible signal examples:

- probability of upward movement,
- probability of downward movement,
- confidence-adjusted trend score,
- class margin between bullish and bearish probabilities.

### Volatility Prediction

The volatility prediction head is not necessarily a directional alpha signal, but it is still useful in alpha research. Predicted volatility can support:

- position sizing,
- risk control,
- regime detection,
- trade filtering,
- confidence adjustment for directional signals,
- market-making or volatility-aware strategies.

For example, a directional trend signal may be more useful when paired with a volatility forecast that controls exposure during unstable regimes.

## Suggested Research Framing

A concise way to describe the idea in the report is:

> The proposed framework does not treat extracted features as alpha factors directly. Instead, it learns reusable market representations from OHLCV sequences and uses lightweight downstream heads to generate task-specific predictive signals. These signals, such as predicted trend, future price movement, or expected volatility, can be interpreted as candidate alpha signals because they indicate potentially exploitable future market behavior. A full alpha claim would require additional validation through out-of-sample signal tests, robustness checks, and trading-oriented evaluation.

This framing keeps the contribution aligned with the current implementation. The main claim remains representation learning and transferable signal generation, while alpha discovery is positioned as a natural downstream use case.

## Relationship to the Existing Task Evaluation

The current evaluation tasks use standard machine learning metrics:

- Price prediction: MAE and RMSE
- Volatility prediction: MSE and Pearson correlation
- Trend classification: Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix

These are appropriate for evaluating supervised task performance. The current budget is limited to the three downstream task evaluations. Alpha mining is a future add-on after the required predictive heads produce aligned, frozen predictions; because the current test split is used for task evaluation, that future add-on requires a fresh holdout or temporally later data for its final factor evaluation.

### 1. Information Coefficient

Measure the correlation between the predicted signal and the future realized target across contracts or windows.

- Pearson IC: linear correlation between signal and future return.
- Spearman rank IC: rank correlation between signal and future return.

Rank IC is often useful when the exact predicted value is less important than whether the model ranks opportunities correctly.

### 2. Signal Quantile Analysis

Sort samples by predicted signal strength and split them into quantiles, such as top 20%, middle 60%, and bottom 20%. Then compare future outcomes across groups.

Useful checks:

- Does the top signal bucket have better future returns than the bottom bucket?
- Is there a monotonic relationship from low signal to high signal?
- Does the signal remain useful across different timeframes?

This is a simple way to test whether the model output behaves like a usable alpha signal.

### 3. Directional Hit Rate

For price or trend signals, evaluate whether the predicted direction matches the realized direction.

This is related to classification accuracy, but can be adapted to trading-style decisions by ignoring weak or low-confidence predictions.

Example:

```text
Only evaluate predictions where confidence > threshold.
Then measure directional accuracy on this filtered subset.
```

This tests whether stronger signals are more reliable than weaker signals.

### 4. Spread Between Strong Positive and Strong Negative Signals

For directional signals, compare the realized future return of high-score samples against low-score samples.

```text
Signal spread = mean future return of top bucket - mean future return of bottom bucket
```

A positive and stable spread suggests that the signal may separate favorable and unfavorable opportunities.

### 5. Volatility-Aware Signal Evaluation

For volatility prediction, evaluate whether the forecast improves downstream decision quality rather than only forecast accuracy.

Possible checks:

- Does predicted volatility identify high-risk regimes?
- Does filtering out high-volatility periods improve trend signal reliability?
- Does volatility-based sizing reduce drawdown in a simple backtest?

This positions volatility forecasting as a risk-aware alpha research component rather than a standalone directional signal.

### Simple Backtest as a Later Extension

A later extension could convert downstream signals into a simple trading rule:

```text
If predicted trend probability > upper threshold: long / buy
If predicted trend probability < lower threshold: short / avoid / sell
Otherwise: no trade
```

For Polymarket event contracts, the exact trading rule would need to respect contract mechanics, liquidity, fees, and whether short exposure is available. This should be treated as a future extension, not as part of the current core framework unless explicitly implemented.

## Recommended Claim Boundary

The safest and clearest claim is:

- The framework learns reusable market representations.
- These representations support lightweight downstream heads.
- The downstream heads generate predictive signals.
- These predictive signals are economically meaningful primitives for symbolic alpha research.
- A selected formula is a candidate factor, not a proven profitable alpha or a trading strategy, without robust held-out signal tests and trading-oriented validation.

The project therefore sits between representation learning and alpha research: it does not directly deliver a production alpha strategy, but it can help generate useful signals that make alpha discovery more systematic.
