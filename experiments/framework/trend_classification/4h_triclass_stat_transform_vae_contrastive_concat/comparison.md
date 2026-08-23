# Framework Trend Classification Comparison

This file records the first framework MVP trend-classification run using the
TA-MLP-style tri-class label: BUY / HOLD / SELL. All numbers are
characterization results on the locked test split; no checkpoint is selected
from test performance.

## Framework Result

Run:

```text
experiments/framework/trend_classification/4h_triclass_stat_transform_vae_contrastive_concat/
```

Configuration:

- task: trend classification
- timeframe: 4h
- sequence length: 64
- branches: `statistical`, `transformed`, `vae`, `contrastive`
- aggregation: concat
- head: `TrendClassifier(n_classes=3)`
- loss: CrossEntropyLoss
- labels: `data/task_labels/trend_classification/triclass_4h_seq64_top50.npz`
- label mode: TA-MLP-style triclass, `BUY=0`, `HOLD=1`, `SELL=2`
- branch standardization: train-fitted z-score with fixed clipping at +/-10

| Epochs | Accuracy | Macro-F1 | Weighted-F1 | Train CE |
|---:|---:|---:|---:|---:|
| 15 | 0.4730 | 0.3875 | 0.4530 | 0.4116 |
| 50 | 0.4953 | 0.4164 | 0.4812 | 0.3890 |
| 100 | 0.5071 | 0.4232 | 0.4900 | 0.3702 |

## Label Bundle

The label bundle is aligned to the processed feature rows through saved
`train_indices` and `test_indices`. Labels are computed at the end timestamp of
each processed sequence window. The final `f_window=2` rows of each contract
split are dropped so labels do not need future rows outside that split.

| Split | Label rows | BUY | HOLD | SELL |
|---|---:|---:|---:|---:|
| train | 109,741 | 9,610 | 91,149 | 8,982 |
| test | 27,400 | 6,764 | 13,827 | 6,809 |

## Baseline Context

Majority-HOLD baseline on this exact framework label bundle:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| always HOLD | 0.5046 | 0.2236 |

Existing TA-MLP triclass benchmark:

| Epochs | Accuracy | Macro-F1 | F1 BUY | F1 HOLD | F1 SELL |
|---:|---:|---:|---:|---:|---:|
| 15 | 0.7116 | 0.4546 | 0.2974 | 0.8625 | 0.2039 |
| 20 | 0.7256 | 0.4524 | 0.3029 | 0.8647 | 0.1897 |
| 25 | 0.7115 | 0.4651 | 0.2936 | 0.8587 | 0.2429 |
| 50 | 0.7118 | 0.4570 | 0.2881 | 0.8542 | 0.2285 |
| 100 | 0.7065 | 0.4595 | 0.2674 | 0.8457 | 0.2654 |

The TA-MLP numbers use the same tri-class labeling formula and training-only
threshold fitting principle, but they are not yet a strict one-to-one comparison
against this framework run. TA-MLP operates on TA-feature rows after indicator
warm-up, while this framework run labels processed sequence windows and keeps
labels inside each processed split. Treat TA-MLP here as external benchmark
context until both models consume a shared saved label bundle.

## Interpretation

The framework trend MVP runs end to end and produces valid tri-class test
metrics. Accuracy is close to the majority-HOLD baseline, so accuracy alone is
not a strong signal. Macro-F1 is much higher than majority-HOLD, which means the
framework is making non-trivial BUY/SELL predictions rather than collapsing to
the largest class.

Across the fixed-budget sweep, train CE decreases monotonically and macro-F1
improves from 0.3875 to 0.4232. This is a useful MVP result, but not yet a
superiority claim over TA-MLP. The next fair-comparison step is to refit TA-MLP
or a TA-feature MLP against the same saved `triclass_4h_seq64_top50.npz` label
bundle so sample counts, labels, and metrics are exactly shared.

Next evaluation steps:

- add BYOL features to the framework feature bundle and rerun this task
- run a Raw-OHLCV MLP tri-class baseline using the same saved label bundle
- refit TA-MLP to consume the shared label bundle or document its remaining row
  alignment differences
- report macro-F1, per-class recall/F1, and confusion matrices as primary trend
  metrics; keep accuracy as supporting context only
- run multi-seed confirmation before making stronger trend-classification claims
