# Phase-1 Trend Classification Comparison

This report compares the Phase-1 five-branch trend classifier with the earlier four-branch framework run, an exact majority-HOLD reference, and the existing TA-MLP benchmark. All epoch budgets are retained as a locked-test characterization sweep; no checkpoint is selected from test performance.

## Phase-1 result

| Epoch | Accuracy | Macro-F1 | Weighted-F1 | Train CE |
|---:|---:|---:|---:|---:|
| 15 | 0.455000 | 0.375672 | 0.438402 | 0.408571 |
| 50 | 0.476095 | 0.394168 | 0.459460 | 0.385257 |
| 100 | 0.476460 | 0.393482 | 0.459768 | 0.363737 |

The run uses the frozen `statistical` (70 dimensions), `transformed` (55), `vae` (64), `contrastive` (128), and `byol` (128) branches. Their 445-dimensional concatenation is standardized using training rows only and passed to the default three-class `TrendClassifier` head.

## Comparison contract

The comparison with the earlier four-branch framework is strict. Both runs use:

- `data/processed/market_4h_seq64_top50.npz` and its chronological per-contract 80/20 split;
- `data/task_labels/trend_classification/triclass_4h_seq64_top50.npz` and the same aligned 109,741 training and 27,400 test rows;
- the same BUY/HOLD/SELL class mapping, train-fitted thresholds, concat aggregation, task-head architecture, feature standardization, seed (`0`), batch size, learning rate, epoch budgets, and metrics.

The controlled configuration difference is the inclusion of the frozen 128-dimensional BYOL branch in Phase-1. The TA-MLP comparison is contextual only: it uses the same broad labeling formula and training-only threshold principle, but operates on a different set of rows after technical-indicator warm-up and has a different test-class distribution.

## Label distribution and majority reference

| Split | Rows | BUY | HOLD | SELL |
|---|---:|---:|---:|---:|
| Train | 109,741 | 9,610 (8.8%) | 91,149 (83.1%) | 8,982 (8.2%) |
| Test | 27,400 | 6,764 (24.7%) | 13,827 (50.5%) | 6,809 (24.9%) |

| Reference | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Always HOLD | 0.504635 | 0.223591 | 0.338496 |

The train-to-test change in class proportions is substantial. Accuracy must therefore be interpreted with macro-F1 and class-level metrics. Phase-1 is below always-HOLD accuracy at every budget, but exceeds its macro-F1 by approximately 0.152–0.171 and its weighted-F1 by 0.100–0.121. The classifier is making non-trivial BUY and SELL predictions instead of collapsing entirely to HOLD, although those predictions introduce enough errors to reduce overall accuracy.

## Strict comparison with the four-branch framework

| Epoch | Phase-1 Accuracy | Four-branch Accuracy | Accuracy change | Phase-1 Macro-F1 | Four-branch Macro-F1 | Macro-F1 change |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.455000 | 0.473029 | -0.0180 | 0.375672 | 0.387460 | -0.0118 |
| 50 | 0.476095 | 0.495255 | -0.0192 | 0.394168 | 0.416447 | -0.0223 |
| 100 | 0.476460 | 0.507117 | -0.0307 | 0.393482 | 0.423161 | -0.0297 |

Phase-1 is worse than the four-branch framework at all three matched budgets. The difference also grows with training duration. Weighted-F1 falls by 0.0146, 0.0218, and 0.0302 at 15, 50, and 100 epochs respectively. Under this concat configuration, adding the current BYOL embedding does not improve trend classification and appears to introduce redundant, weak, or poorly aligned features for the supervised head.

This result is evidence about the current BYOL-plus-concat configuration, not a general conclusion that BYOL cannot represent trend information. A gated aggregator, branch projection, different BYOL backbone, or independently tuned downstream probe could produce different behavior. Multiple seeds are also required before treating these single-seed differences as stable.

## Class-level behavior

| Epoch | BUY Precision | BUY Recall | BUY F1 | HOLD Precision | HOLD Recall | HOLD F1 | SELL Precision | SELL Recall | SELL F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.2593 | 0.4026 | 0.3154 | 0.5985 | 0.6435 | 0.6202 | 0.4163 | 0.1242 | 0.1914 |
| 50 | 0.2662 | 0.3380 | 0.2978 | 0.6086 | 0.6939 | 0.6485 | 0.3823 | 0.1710 | 0.2362 |
| 100 | 0.3197 | 0.2794 | 0.2982 | 0.6050 | 0.7061 | 0.6517 | 0.2620 | 0.2059 | 0.2306 |

HOLD is consistently the easiest class. SELL is the hardest, with recall between 0.124 and 0.206. Longer training increases HOLD recall and initially improves SELL recall, but BUY recall falls from 0.403 at 15 epochs to 0.279 at 100 epochs. The model is redistributing errors across minority classes rather than producing a broad improvement.

The 50-epoch confusion matrix shows the main failure pattern:

| Actual / predicted | BUY | HOLD | SELL |
|---|---:|---:|---:|
| BUY | 2,286 | 3,208 | 1,270 |
| HOLD | 3,621 | 9,595 | 611 |
| SELL | 2,682 | 2,963 | 1,164 |

Large numbers of BUY and SELL rows are predicted as HOLD, while many true HOLD rows are predicted as BUY. This is consistent with a classifier trained on an 83%-HOLD training distribution and evaluated on a test distribution in which HOLD accounts for only about half the rows.

## Training-duration behavior

From 15 to 50 epochs, accuracy increases by 0.0211 and macro-F1 by 0.0185. From 50 to 100 epochs, accuracy changes by only 0.0004 and macro-F1 decreases by 0.0007, even though training cross-entropy continues falling from 0.3853 to 0.3637. Test performance has therefore plateaued by the longer budgets, with mild train–test divergence. The complete sweep should be reported; the 50-epoch result must not be selected retrospectively because its macro-F1 is highest on the locked test set.

## TA-MLP benchmark context

| Epoch | Phase-1 Accuracy | TA-MLP Accuracy | Phase-1 Macro-F1 | TA-MLP Macro-F1 |
|---:|---:|---:|---:|---:|
| 15 | 0.4550 | 0.7116 | 0.3757 | 0.4546 |
| 50 | 0.4761 | 0.7118 | 0.3942 | 0.4570 |
| 100 | 0.4765 | 0.7065 | 0.3935 | 0.4595 |

TA-MLP has higher reported accuracy and macro-F1, but the accuracy gap is especially unsuitable for a direct claim because TA-MLP's test data is 74.7% HOLD, compared with 50.5% HOLD in the shared framework label bundle. Its 16,258 test rows also differ from the framework's 27,400 aligned rows. TA-MLP should remain external characterization evidence until it is rerun using the saved framework label bundle and exact aligned feature rows.

## Conclusion

The Phase-1 representation contains usable trend information: its macro-F1 is meaningfully above the exact majority-HOLD reference, and it predicts all three classes. However, its accuracy remains below the majority reference, minority-class recall is weak, and performance plateaus after 50 epochs. The strict matched comparison also shows that adding the current BYOL branch to the four-branch concat representation reduces accuracy, macro-F1, and weighted-F1 at every tested budget.

The defensible conclusion is that the five-branch Phase-1 trend probe works end to end but does not improve on the earlier four-branch framework for this task. It also does not yet outperform the TA-MLP benchmark context. No single epoch budget should be presented as a selected model based on these test results.

## Recommended follow-up

- Repeat the matched four-branch and five-branch comparison over multiple seeds and report mean plus standard deviation.
- Run predeclared branch ablations, especially BYOL-only and all-branches-minus-BYOL, to distinguish redundancy from weak standalone BYOL features.
- Inspect per-contract class distributions and errors to determine whether the aggregate train-to-test label shift is concentrated in a small number of contracts.
- Treat class-weighted loss, balanced sampling, gated fusion, or projected branch fusion as new predeclared experiments rather than post-test adjustments to this run.
- Rerun TA-MLP or a TA-feature MLP on the exact saved label rows before making a strict benchmark claim.
