# Phase-1 Experiment Observations and Research Judgement

Date: 2026-09-01

## Purpose

This note records the main observations and research judgement formed after the Phase-1 price-prediction, volatility-prediction, and trend-classification experiments. It separates conclusions supported directly by the recorded results from hypotheses that should be tested in the next experiment stage.

The Phase-1 framework uses the same frozen five-branch representation for all downstream tasks:

| Branch | Dimension |
|---|---:|
| Statistical | 70 |
| Transformed | 55 |
| VAE | 64 |
| Contrastive | 128 |
| BYOL | 128 |
| **Concatenated representation** | **445** |

Concat aggregation has no learnable parameters. The downstream regression head is a lightweight probe with the structure `445 → 128 → 64 → 1`, using two nonlinear hidden layers. In this note, this component is called the **task head**, **decoder**, or **probe**. It is distinct from the projector networks used only during contrastive or BYOL pretraining.

## Researcher intuition

The main intuition is:

> Temporal models such as LSTMs are generally better equipped than static MLPs to model sequential financial dynamics. Despite using only a shallow static task head, the Phase-1 frozen representation already produces reasonably strong regression results, especially at smaller epoch budgets. This suggests that the representation itself is expressive and already contains substantial predictive information. A more capable decoder—particularly one that models temporal evolution—may be able to extract more value from these representations and narrow the remaining gap to stronger task-specific benchmarks.

For trend classification, the intuition is different:

> The weak classification result may be caused partly by the label definition rather than only by representation or classifier capacity. The current BUY/HOLD/SELL algorithm was transferred directly from stock-market technical analysis and may impose thresholds and regimes that are not appropriate for bounded, event-driven prediction-market probabilities.

The results broadly support both intuitions, subject to the qualifications below.

## Observation 1: the regression representations are already informative

### Price prediction

| Epoch | Phase-1 MAE | Raw-OHLCV MLP MAE | Phase-1 RMSE | Raw-OHLCV MLP RMSE |
|---:|---:|---:|---:|---:|
| 15 | 0.051227 | 0.083424 | 0.090831 | 0.106712 |
| 50 | 0.065077 | 0.060006 | 0.099259 | 0.080469 |
| 100 | 0.067843 | 0.045375 | 0.100879 | 0.068346 |

At the matched 15-epoch budget, the frozen five-branch framework reduces MAE by 38.6% and RMSE by 14.9% relative to the Raw-OHLCV MLP. This is important because the framework uses fixed representations plus a shallow probe, while the raw MLP learns its representation and predictor jointly from flattened OHLCV input.

The advantage does not persist at 50 and 100 epochs. The Raw-OHLCV MLP continues improving, while the framework's test error worsens even as its training loss falls. Therefore, the result supports an **early-budget representation-value claim**, not a general claim that the current framework is better than the raw MLP at every training duration.

The existing price LSTM has much lower reported error and supports the broader intuition that sequential modeling is valuable. However, it remains contextual evidence because its stored artifact uses close-only windows and has a one-row target-alignment difference. It should not yet be cited as a strict proof that an LSTM decoder will always beat an MLP under an otherwise identical contract.

### Volatility prediction

| Epoch | Phase-1 MSE | Raw LSTM MSE | GARCH–LSTM stack MSE | Phase-1 correlation | Raw LSTM correlation | Stack correlation |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.007935 | 0.011799 | 0.007926 | 0.767 | 0.690 | 0.806 |
| 50 | 0.007491 | 0.009625 | 0.007483 | 0.770 | 0.677 | 0.810 |
| 100 | 0.007688 | 0.010631 | 0.006951 | 0.764 | 0.640 | 0.822 |

This comparison is strict: Phase-1, Raw LSTM, and the adapted GARCH–LSTM stack use exactly identical 27,450 saved test targets. Phase-1 reduces MSE by approximately 22–33% relative to Raw LSTM at every budget. At 15 and 50 epochs, its RMSE and MSE are almost identical to those of the stronger GARCH–LSTM stack, although the stack retains lower MAE and higher correlation.

This is strong evidence that the representation contains useful volatility structure. The static shallow probe is competitive with a substantially more specialized hybrid model on squared error at the shorter budgets. It does not beat the stack overall, but its proximity suggests that downstream extraction capacity and fusion design are plausible remaining bottlenecks.

## Judgement on regression representation quality

The combined regression evidence supports the following judgement:

1. The representations are not merely compressed inputs with little downstream value. A shallow probe extracts enough information to beat a raw MLP at the short price budget and to beat Raw LSTM consistently on the strict volatility task.
2. The framework's useful performance across two different regression targets supports representation transferability: the same frozen feature branches work for both price and volatility without retraining the encoders.
3. The current decoder is probably not extracting all available information. It treats the 445-dimensional representation as one static vector and cannot model how representations evolve across consecutive windows.
4. A stronger decoder may improve performance, but simply adding MLP depth is not guaranteed to help. Existing train–test divergence shows that extra capacity could increase overfitting.
5. The more precise next hypothesis is therefore not “a deeper MLP will be better,” but “branch-aware and temporal decoding can extract predictive structure that a shallow static probe cannot.”

The statement that temporal models **definitely** beat static MLPs should remain an intuition rather than a final empirical conclusion. The price LSTM evidence is not yet strictly aligned, and the Phase-1 volatility framework already beats the Raw LSTM. Temporal architecture is often advantageous, but its benefit depends on input construction, target definition, optimization, and decoder capacity.

## Proposed decoder direction

The next decoder study should isolate three different sources of improvement:

| Configuration | Purpose |
|---|---|
| Current concat + shallow MLP | Preserve the Phase-1 reference probe |
| Branch projections + residual MLP | Test modest additional nonlinear capacity and cross-branch interactions |
| Gated aggregator + same shallow MLP | Test whether task-dependent branch weighting is more important than head depth |
| Temporal decoder over consecutive frozen embeddings | Test whether sequential evolution of the representations adds information |

A genuinely temporal decoder requires an ordered embedding sequence rather than a single vector:

```text
K consecutive OHLCV windows
        ↓
frozen five-branch representation for each window
        ↓
[batch, K, representation_dim]
        ↓
small GRU, LSTM, TCN, or Transformer
        ↓
task prediction
```

Applying an LSTM to a single 445-dimensional vector would not provide meaningful temporal modeling. The data pipeline must preserve contract identity, chronological order, split boundaries, and horizon-safe alignment when building sequences of frozen embeddings.

Task-aware output constraints should also be included in the decoder design:

- Price prediction can test a bounded probability output or residual probability-change formulation.
- Volatility prediction should use a predeclared nonnegative output such as Softplus. The current unconstrained head produced 5.8–8.4% negative predictions; diagnostic clipping helped only slightly but confirmed the architectural mismatch.
- A log-volatility target may be tested as a separate predeclared experiment because volatility is nonnegative and strongly right-skewed.

## Observation 2: the trend label appears mismatched to prediction markets

The current labels reproduce the TA-MLP stock-market algorithm:

- backward EWM span `b_window=5`;
- forward horizon `f_window=2`;
- `alpha` and `beta` fitted from training percentage-change quantiles;
- BUY or SELL only when movement lies between the lower and upper thresholds;
- movements below `alpha` and extreme movements above the effective `beta` cap are assigned HOLD.

This design is questionable for prediction markets. Large probability jumps often represent important information arrival, yet the upper-cap rule may classify them as HOLD. Percentage changes are also unstable near probability zero, while prediction-market prices are bounded and converge toward zero or one near resolution.

The saved label bundle shows a large distribution shift:

| Split | BUY | HOLD | SELL |
|---|---:|---:|---:|
| Train | 8.8% | 83.1% | 8.2% |
| Test | 24.7% | 50.5% | 24.9% |

Phase-1 accuracy remains below the exact majority-HOLD reference, although macro-F1 is meaningfully higher. This means the representation predicts non-trivial minority-class structure, but the target regimes themselves change substantially between the chronological training and test periods.

Adding BYOL through concat fusion also reduces trend accuracy and macro-F1 relative to the strictly matched four-branch run. Classification weakness therefore has at least two plausible sources: an unsuitable label contract and unhelpful or redundant branch information under naive concatenation. A new label alone should not be assumed to solve every classification problem.

## Judgement on classification

The current trend task should be retained as a **stock-label transfer experiment**: it demonstrates what happens when an established technical-analysis labeling method is applied to prediction-market data. It should not automatically be treated as the definitive prediction-market regime task.

A more domain-appropriate initial target is absolute probability movement:

\[
\Delta p_{t,h}=p_{t+h}-p_t.
\]

The classes can then be defined as:

- UP when \(\Delta p_{t,h}>\tau\);
- DOWN when \(\Delta p_{t,h}< -\tau\);
- STABLE otherwise.

The threshold \(\tau\) should be fixed in probability points or fitted from training rows only. It may also be scaled by training-fitted local volatility. An alternative is to predict \(\Delta p\) as a regression target first and derive the three classes from predeclared thresholds, preserving more information than direct rigid classification.

A later regime task could distinguish stable trading, gradual repricing, information jumps, and resolution/convergence. Such labels need explicit causal definitions and sufficient class support before they are used for comparison.

## Experimental interpretation boundary

The Phase-1 results support the claim that frozen multi-branch features contain useful downstream information. They do not yet prove that:

- a temporal decoder will outperform every static decoder;
- increased decoder capacity alone will close the benchmark gap;
- BYOL is generally harmful rather than unhelpful under the current concat probe;
- the proposed probability-movement labels are superior before they are implemented and evaluated;
- the framework is universally better than task-specific baselines.

Because these follow-up designs were motivated partly by observed locked-test results, they should be recorded as a new exploratory experiment stage. Reusing the current test split can provide characterization evidence, but a strong final claim requires a fresh later temporal holdout or an explicitly revised evaluation protocol.

## Recommended next experiment sequence

1. Define a small, predeclared decoder matrix: current MLP, residual branch-aware MLP, gated fusion, and one compact temporal decoder over consecutive embeddings.
2. Correct task-output constraints, particularly a nonnegative volatility head.
3. Implement a prediction-market probability-movement label bundle with train-only threshold fitting and split-safe horizons.
4. Rerun framework and relevant baselines on the exact same new label rows; do not compare raw accuracy directly between different label definitions.
5. Run branch and leave-one-branch-out ablations so decoder gains are not confused with representation composition.
6. Use multiple seeds and paired uncertainty estimates before making stronger claims.
7. Reserve a fresh temporally later holdout for final confirmation because the current locked-test results informed these follow-up hypotheses.

## Overall conclusion

The regression results are encouraging precisely because they were obtained with a deliberately simple probe. They indicate that the multi-branch representation is expressive and transferable enough to support useful predictions without end-to-end encoder tuning. A branch-aware or genuinely temporal decoder is therefore a well-motivated next experiment, provided that it is evaluated as a decoder-capacity hypothesis rather than automatically attributed to representation quality.

The classification result points more strongly toward task-definition revision. The transferred stock-market labeling rule does not align cleanly with the bounded, event-driven behavior of prediction-market probabilities, and its train/test class shift is substantial. Redesigning the classification target around absolute probability movement or prediction-market regimes is likely more valuable than only increasing classifier depth.

## Supporting experiment reports

- [Phase-1 price comparison](../../experiments/framework/phase1/price_prediction/4h_phase1_all5_concat/comparison.md)
- [Phase-1 volatility comparison](../../experiments/framework/phase1/volatility_prediction/4h_phase1_all5_concat/comparison.md)
- [Phase-1 trend comparison](../../experiments/framework/phase1/trend_classification/4h_phase1_all5_concat/comparison.md)
