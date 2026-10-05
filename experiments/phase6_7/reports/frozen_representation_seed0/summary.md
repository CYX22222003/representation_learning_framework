# Phase 6.7 frozen-representation baseline comparison

**Date:** 2026-10-05  
**Scope:** seed 0, epoch 50, two frozen walks, identical task rows, and the
same task-specific native-width probe family.

## Comparison boundary

The precommitted direct comparison is between immutable canonical `H0` and
the three independently adapted external representations: SaURL-TS-Frozen,
LWA-Frozen, and TimeDART-Frozen. The tables also show a **best internal
encoder-variant envelope** assembled from completed Phase 6 and Phase 6.5
experiments. That envelope is descriptive and metric-wise: different variants
can supply different cells, so it is not one selected deployable model and was
not chosen before Phase 6.7 results were observed.

All conclusions below concern frozen-representation transfer under this
project's protocol. They do not establish state of the art, statistical
significance, profitable trading, or superiority under the source papers'
original fine-tuning protocols.

## Principal result

The framework's clearest advantage is eight-hour future-price prediction.
The best internal encoder variant in each walk has lower MAE than every
external representation by **24.9% to 52.6%**, and lower RMSE by **5.7% to
41.8%**. Even immutable `H0` has lower price MAE than all three external
representations in both walks.

TimeDART is the strongest method for two-hour trend classification. Its lead
over the best observed internal encoder variant is modest: `0.80`--`1.96`
percentage points in macro-F1 and `0.35`--`1.87` percentage points in balanced
accuracy. The best internal variants remain ahead of SaURL-TS and LWA on both
classification metrics in both walks, although the margins range from nearly
tied to several percentage points.

For eight-hour realised-variance prediction, TimeDART has the lowest MAE in
both walks. Its MAE is only **2.6%** lower than the best internal encoder
variant in Walk 1 and **4.6%** lower in Walk 2. The best internal variant, in
turn, has **2.3% to 5.8%** lower MAE than SaURL-TS and LWA. RMSE differences
among these methods are below `0.6%` and change ordering across walks, so the
RMSE evidence is best described as effectively tied rather than as a stable
win for any method.

## Eight-hour future-price prediction

Lower MAE and RMSE are better.

| Method | Walk 1 MAE | Walk 1 RMSE | Walk 2 MAE | Walk 2 RMSE |
|---|---:|---:|---:|---:|
| Best internal encoder variant | **0.006933** (`HC-AL`) | **0.015184** (`HC-AL`) | **0.007888** (`HB-AL`) | **0.024590** (`HB-AL`) |
| Canonical `H0` | 0.007985 | 0.018906 | 0.010054 | 0.030073 |
| SaURL-TS-Frozen | 0.009226 | 0.017709 | 0.013524 | 0.031302 |
| LWA-Frozen | 0.012594 | 0.023817 | 0.015669 | 0.028236 |
| TimeDART-Frozen | 0.014638 | 0.026109 | 0.013745 | 0.026081 |

Relative error reduction of the best internal variant against each external
representation:

| External representation | Walk 1 MAE | Walk 1 RMSE | Walk 2 MAE | Walk 2 RMSE |
|---|---:|---:|---:|---:|
| SaURL-TS-Frozen | 24.9% | 14.3% | 41.7% | 21.4% |
| LWA-Frozen | 44.9% | 36.2% | 49.7% | 12.9% |
| TimeDART-Frozen | 52.6% | 41.8% | 42.6% | 5.7% |

This supports a substantial price-prediction advantage over the three recent
frozen-representation baselines, particularly on MAE. It does not make the
framework the best forecasting system already tested in the project: Raw LSTM
and persistence remain stronger price-error references under their respective
contracts.

## Two-hour trend classification

Higher macro-F1 and balanced accuracy are better.

| Method | Walk 1 macro-F1 | Walk 1 balanced accuracy | Walk 2 macro-F1 | Walk 2 balanced accuracy |
|---|---:|---:|---:|---:|
| Best internal encoder variant | 0.453953 (`HB-SL2`) | 0.469034 (`HB-SL2`) | 0.458697 (`HB-ST`) | 0.495388 (`HC-ST`) |
| Canonical `H0` | 0.453469 | 0.465787 | 0.453202 | 0.489162 |
| SaURL-TS-Frozen | 0.429292 | 0.463001 | 0.450317 | 0.488291 |
| LWA-Frozen | 0.403129 | 0.468943 | 0.449673 | 0.483677 |
| TimeDART-Frozen | **0.473548** | **0.487778** | **0.466718** | **0.498915** |

TimeDART therefore wins this task in both walks, but only by a small margin
over the strongest internal encoder variants. The framework still compares
favorably with SaURL-TS and LWA. Because the internal row is a metric-wise
envelope and all results are seed 0, this should be reported as comparative
evidence rather than a definitive ranking.

## Eight-hour realised-variance prediction

Lower MAE and RMSE are better. Correlation metrics are intentionally omitted
from this headline comparison; the interpretation rests on conventional
regression error.

| Method | Walk 1 MAE | Walk 1 RMSE | Walk 2 MAE | Walk 2 RMSE |
|---|---:|---:|---:|---:|
| Best internal encoder variant | 0.000625186 (`HB-ST`) | **0.025581775** (`HB-SL2`) | 0.000504746 (`HC-AL`) | 0.009857093 (`HB-AT`) |
| Canonical `H0` | 0.000626190 | 0.025583491 | 0.000502040 | 0.009858725 |
| SaURL-TS-Frozen | 0.000647110 | 0.025586692 | 0.000535907 | 0.009821329 |
| LWA-Frozen | 0.000654246 | 0.025588726 | 0.000516424 | **0.009807731** |
| TimeDART-Frozen | **0.000608884** | 0.025584718 | **0.000481353** | 0.009848405 |

On MAE, TimeDART has a small but consistent advantage over the strongest
internal variants, while the internal variants have a small consistent
advantage over SaURL-TS and LWA. RMSE does not reproduce that ordering: the
best internal variant is marginally best in Walk 1, LWA is marginally best in
Walk 2, and all gaps are very small. The defensible conclusion is therefore a
small TimeDART MAE advantage, not broad volatility-regression dominance.

## Overall judgement

The results show task-dependent specialization rather than one universal
winner:

- the unified multi-branch framework is substantially stronger than all three
  external frozen representations for future-price error, with the clearest
  and most consistent separation on MAE;
- TimeDART is modestly stronger than the best observed internal variants for
  trend classification;
- TimeDART also has a small realised-variance MAE advantage, while volatility
  RMSE is effectively tied and mixed;
- the best internal variants generally outperform SaURL-TS and LWA on trend
  classification and volatility MAE; and
- these are two-walk, seed-0 frozen-probe results. Multiple-seed confirmation
  and fresh data would be required before making stronger generalization or
  SOTA claims.

## Evidence sources

- External method metrics:
  `experiments/phase6_7/downstream/<task>/<method>/walk<k>/seed0/e50/metrics.json`
- Phase 6 encoder variants:
  `experiments/phase6/encoder_variants/reports/complete_seed0/epoch50_by_walk.csv`
- Phase 6.5 LSTM-capacity variants:
  `experiments/phase6_5/lstm_capacity/reports/complete_seed0/epoch50_by_walk.csv`
- Phase 6.5 residual-CNN variants:
  `experiments/phase6_5/residual_cnn/reports/complete_seed0/epoch50_by_walk.csv`
- Method-local summaries:
  `experiments/phase6_7/reports/saurl_staged_seed0/summary.md` and
  `experiments/phase6_7/reports/lwa_staged_seed0/summary.md`
