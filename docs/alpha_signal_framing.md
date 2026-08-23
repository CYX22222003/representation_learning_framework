# Food for Thought: Alpha-Signal Framing and Alternative Evaluation Ideas

This note is not an implementation guide. It is a conceptual framing for how the learned representations in this project can be discussed in an alpha research or alpha validation context.

## Core Idea

The framework should not be described as directly producing alpha factors from the raw feature branches. A more precise framing is:

```text
Raw OHLCV sequence
  -> multi-branch market representations
  -> lightweight downstream task head
  -> predictive signal
  -> candidate alpha signal or research input
```

In this framing, the statistical, transformed, VAE, contrastive, and BYOL branches are reusable market representations. They encode information about price dynamics, volatility structure, frequency patterns, nonlinear temporal structure, and regime behavior. The downstream task head then converts those representations into task-specific outputs, such as predicted price movement, predicted volatility, or trend probability.

Those task-head outputs are the objects that are closest to alpha signals. They are not guaranteed alpha factors by definition, but they can be treated as candidate alpha signals if they show reliable out-of-sample predictive power for future market behavior.

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

## Alternative Evaluation Ideas

The current evaluation tasks use standard machine learning metrics:

- Price prediction: MAE and RMSE
- Volatility prediction: MSE and Pearson correlation
- Trend classification: Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix

These are appropriate for evaluating supervised task performance. The current MVP should focus on completing the three downstream tasks, BYOL feature extraction, and branch ablations first. If the project later wants to connect more directly to alpha research, the following optional evaluations could be added.

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

### 6. Simple Backtest as an Extension

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
- These predictive signals can be used as candidate alpha research inputs.
- A complete alpha factor or trading strategy claim requires additional trading-oriented validation.

The project therefore sits between representation learning and alpha research: it does not directly deliver a production alpha strategy, but it can help generate useful signals that make alpha discovery more systematic.
