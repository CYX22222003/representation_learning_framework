# Final Research Objectives: Representation Learning for Alpha-Related Signal Generation

This document summarises the final research positioning for the project. It is intended as report-writing guidance, not as an implementation guide.

## Final Research Aim

The main aim of this project is to investigate whether a unified multi-branch representation learning framework can generate reusable market representations from OHLCV time-series data, and whether those representations can support useful downstream predictive signals for alpha-related research tasks.

The project should not be framed as directly discovering profitable alpha factors or producing a complete trading strategy. Instead, the safer and more accurate framing is:

```text
Raw OHLCV sequences
  -> reusable market representations
  -> lightweight downstream heads
  -> predictive signals
  -> candidate alpha research inputs
```

In this view, the framework is a representation engine. The downstream task heads convert the learned representations into signals such as predicted price movement, expected volatility, or trend probability. These signals may be interpreted as candidate alpha signals only after out-of-sample validation.

## Core Research Objectives

1. **Build a transferable representation framework for market time series.**

   The framework combines statistical features, transformation-based features, and self-supervised neural embeddings into a unified representation. This addresses the limitation of models that rely on only one feature family or are trained for only one task.

2. **Evaluate whether frozen representations make downstream tasks easier.**

   The framework uses frozen encoders and lightweight task heads. If the representations are useful, a simple decoder should be able to extract predictive information for price prediction, volatility prediction, and trend classification.

3. **Compare against fair internal and external baselines.**

   The framework is compared against raw OHLCV baselines, single-branch ablations, classical/statistical baselines, and task-specific benchmark models. All models should use the same chronological train/test split, fixed epoch budgets, and consistent metrics.

4. **Analyse which representation branches are useful for which tasks.**

   The goal is not only to show that the full framework performs well, but also to understand whether statistical, transformed, VAE, contrastive, or BYOL representations contribute differently across tasks.

5. **Position downstream predictions as candidate alpha signals.**

   The task-head outputs can be interpreted as predictive signals relevant to alpha research. Price and trend predictions are closer to directional alpha signals, while volatility predictions are useful for risk control, regime detection, filtering, and position sizing.

## Main Research Questions

The project can be evaluated through the following research questions:

1. Do multi-branch representations improve downstream prediction compared with raw OHLCV baselines?
2. Which representation branches contribute most to price, volatility, and trend tasks?
3. Does aggregating multiple branches outperform single-branch representations?
4. Can frozen representations with simple task heads remain competitive with end-to-end task-specific models?
5. Do the downstream task outputs behave like useful candidate alpha signals?
6. Where does the framework fail, and what does that reveal about OHLCV-only representation learning?

These questions make the project robust to uncertain results. The paper does not depend entirely on the full framework beating every benchmark.

## Result Interpretation Strategy

The final report should be prepared to explain several possible result outcomes.

### If the framework beats Raw-OHLCV MLP but not task-specific benchmarks

This is still a meaningful result. It suggests that the learned representations add value over naive raw-sequence learning, while specialised architectures may still be stronger for their own tasks.

Suitable interpretation:

> The framework improves over simple raw OHLCV baselines, indicating that the learned representations encode useful predictive structure. However, task-specific models remain competitive, suggesting that specialised architectures still have advantages for individual forecasting tasks.

### If some branches work better than others

This is a useful ablation result. It shows that different representation families capture different market properties.

Suitable interpretation:

> The ablation study shows that representation usefulness is task-dependent. Statistical and volatility-sensitive features may be more useful for volatility forecasting, while neural embeddings may be more useful for trend or price-movement prediction.

### If full aggregation is not always better than single branches

This does not invalidate the project. It suggests that naive fusion can introduce noise, or that some tasks require branch selection rather than using every representation.

Suitable interpretation:

> Multi-branch fusion is not universally beneficial. The results suggest that future work should explore task-aware branch selection, improved gating, or regularisation to prevent weaker branches from diluting stronger task-specific signals.

### If the framework performs similarly to benchmarks

Comparable performance can still support the representation-learning claim, especially because the framework uses frozen encoders and lightweight task heads.

Suitable interpretation:

> Comparable performance from frozen representations and simple task heads suggests that the learned embeddings preserve useful downstream information, while requiring less task-specific retraining than fully end-to-end benchmark models.

### If the framework underperforms

This is still reportable if the comparison is fair and the analysis is honest. The contribution becomes diagnostic rather than performance-superiority based.

Suitable interpretation:

> The results suggest that OHLCV-only reusable representations may be insufficient for strong alpha-related prediction on Polymarket event contracts. This motivates richer inputs such as order book information, liquidity features, event metadata, news, or market microstructure signals.

## Recommended Claim Boundary

The strongest defensible claim is:

> This project investigates whether unified time-series representations can support alpha-related predictive signal generation in event prediction markets. Rather than directly proposing a trading strategy, it evaluates whether frozen multi-branch embeddings make downstream forecasting and classification tasks easier for lightweight heads.

Avoid claiming:

- The framework discovers profitable alpha.
- The learned features are automatically alpha factors.
- The model is a complete trading system.
- Strong trading performance is proven without backtesting, cost modelling, and liquidity analysis.

Prefer claiming:

- The framework learns reusable market representations.
- The representations support downstream predictive signal generation.
- The task-head outputs can be treated as candidate alpha signals.
- The evaluation tests representation quality through price, volatility, and trend tasks.
- Mixed or negative results still reveal which representations are useful and where OHLCV-only modelling is limited.

## Paper Contribution Even Under Uncertain Results

Even if the framework does not consistently outperform every benchmark, the project can still make a valid contribution by providing:

- a unified multi-branch representation framework for event prediction market time series,
- a fair frozen-encoder probing setup for alpha-related downstream tasks,
- ablation evidence showing which representation families help which tasks,
- comparison against raw OHLCV and task-specific baselines,
- a careful alpha-signal framing that separates representations, predictions, candidate alpha signals, and trading strategies,
- diagnostic evidence about the limitations of OHLCV-only signal generation.

The final paper should therefore be framed as an investigation into transferable representation learning for alpha-related signal generation, not as a guaranteed alpha-discovery system.
