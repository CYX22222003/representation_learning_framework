# Phase 5 Main Experiment Plan

**Date:** 2026-09-21  
**Status:** Core design decisions frozen; implementation and validation pending  
**Predecessor:** `2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`

## 1. Authority and evolution of the research design

This document is the canonical contract for the Phase 5 main experiment. When
the initial `Research_Ideas_Writeup.md`, `research_plan.md`, `design.md`, an
older phase plan, or an earlier data-analysis handoff disagrees with this
document about Phase 5, this document takes precedence.

The project has changed naturally as evidence accumulated. Phases 1--3 exposed
and progressively corrected problems in preprocessing order, contract-boundary
alignment, task definitions, staleness, lifecycle-biased evaluation, and
baseline fairness. Phase 4 was not another failed model experiment: it was the
dedicated data-selection and exploratory-analysis phase, and it launched no
encoder, downstream-head, or baseline training. Phase 5 therefore revises parts
of the initial research idea, research plan, and experiment design before the
main experiments. Future readers must not infer the active Phase 5 contract
from those initial documents without checking this plan.

## 2. Phase 5 objective and four stages

Phase 5 runs the first complete recent-data main experiment under the revised
contract:

1. prepare walk-specific training and evaluation data;
2. train the canonical encoders and downstream models separately per walk;
3. extract frozen features and evaluate probability-movement regression and
   classification; and
4. train matched baselines and compare them on identical rows.

The main research question is whether the canonical five-branch frozen
representation provides useful information for short-horizon probability
movement relative to simple and task-specific alternatives.

## 3. Accepted research assumptions

Phase 5 explicitly adopts the following assumptions as appropriate for the
scope of an undergraduate FYP:

1. Retrospective market selection is acceptable. The fresh Walk 1 and Walk 2
   top-50 cohorts are the primary experiment universes even though FinData
   catalog volume and complete metadata were observed retrospectively.
2. The approved forward-confirmed pruning is assumed to identify the anomalous
   mixed-orientation candles correctly. The final clean candle files are
   treated as the corrected historical series.
3. Anomaly pruning is offline retrospective data cleaning, not a decision-time
   trading signal. Phase 5 does not replay removals according to
   `quarantine_available_at`.
4. Claims are confirmatory within these declared dataset and cleaning
   assumptions. Raw files, audit decisions, and clean-file hashes remain
   preserved for reproducibility.

These assumptions replace the earlier Phase 5 handoff requirements for a fully
cutoff-local catalog universe and decision-time quarantine replay. They do not
relax the model-evaluation boundary: later-walk observations still cannot train
or select an earlier-walk model.

## 4. Primary data and global-calendar walks

The primary source is the final clean native one-hour FinData condition-candle
data in the two independently acquired top-50 walk directories:

```text
data_new/findata/polymarket/
  phase5_walk1_top50_train-2025-12-02_cutoff-2026-04-01_eval-end-2026-06-16/
  phase5_walk2_top50_train-2026-02-16_cutoff-2026-06-16_eval-end-2026-09-01/
```

The primary half-open intervals are:

| Walk | Training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Every cutoff is one global calendar timestamp shared across contracts. Walk 2
may train on observations from the earlier Walk 1 evaluation period because
they are historical by the Walk 2 cutoff. Walk 2 state must never update,
select, or reinterpret the Walk 1 model or its saved predictions.

The fixed December--August top-50/top-100 cohorts remain data-analysis or
sensitivity sources rather than the primary two-walk training data. Native
15-minute data remains an optional resolution sensitivity after the core
one-hour experiment.

## 5. Sequence, gap, and activity contract

- Native resolution: one hour.
- Context length: 64 native bars = 64 hours.
- Fill policy: insert only a complete isolated missing hour.
- Synthetic candle: flat OHLC at the preceding observed close, `volume=0`,
  and `trades=0` when present.
- Longer gaps: remain absent and split sequences.
- Decision endpoint: observed candle only.
- Target endpoint: observed candle only.
- Primary activity eligibility: at least one non-zero close-to-close movement
  in the preceding 24 hourly intervals, computed contract-locally.
- Targets and contexts may never cross a contract, segment, walk-training
  cutoff, or evaluation-end boundary.

The current Walk 1 and Walk 2 filled artifacts satisfy an exact invariant:
every imputed row has zero volume and every observed row has positive volume.
Therefore zero volume is the Phase 5 model's implicit missingness indicator.
The canonical neural and raw-sequence inputs remain the original five OHLCV
channels; no sixth mask or time-since channel is added.

The builder must nevertheless preserve `is_observed`, `is_imputed`,
`original_gap_length_bars`, and `time_since_last_observation` for validation,
row eligibility, and reporting. It must fail if either of these raw-data
invariants is violated before scaling:

```text
is_imputed  -> volume == 0
is_observed -> volume > 0
```

Report metrics separately for contexts that touch an imputed row and contexts
that contain observed rows only. Fitted preprocessing uses the corresponding
walk's training rows only.

The use of an implicit missingness signal is consistent with the principle in
Che et al. (2018), *Recurrent Neural Networks for Multivariate Time Series with
Missing Values*, that a model should be able to distinguish observed and
missing measurements. GRU-D requires explicit masks and time intervals for
arbitrary variable-level missingness; Phase 5 has the simpler whole-candle,
one-bar case with an exact zero-volume sentinel.

## 6. Primary task contract

Both primary tasks use the same two-hour horizon on native one-hour candles:

```text
delta[t, 2h] = close[t + 2 hourly bars] - close[t]
```

Using one horizon gives regression and classification identical decision
times, target times, eligible rows, and gap/activity rules. A different horizon
would define a separate task and must be labelled as a later sensitivity.

### 6.1 Probability-movement regression

- Target: continuous signed two-hour probability change.
- Mandatory reference: predict exact zero movement.
- Primary metrics: MAE, RMSE/MSE, Pearson correlation, Spearman correlation,
  and sign agreement.
- Required breakdowns: each walk, pooled out-of-future predictions,
  row-weighted, contract-macro, lifecycle, non-zero targets,
  threshold-exceeding targets, and imputation exposure.
- Arithmetic-return regression at the same two-hour horizon is secondary and
  requires a frozen zero-price rule and starting-price-band reporting.

### 6.2 Probability-movement classification

The fixed Phase 5 label is:

```text
DOWN   if delta < -0.001
STABLE if abs(delta) <= 0.001
UP     if delta > 0.001
```

The smaller threshold is justified by the shorter two-hour horizon; it is not
chosen merely to force equal class counts. A training-only diagnostic on the
fresh walk cohorts produced approximately:

| Walk | DOWN | STABLE | UP |
|---:|---:|---:|---:|
| 1 | 23.72% | 53.60% | 22.69% |
| 2 | 22.02% | 57.28% | 20.70% |

Use one fixed threshold across both walks so that class meanings remain
comparable. The primary learned-model protocol uses training-prior
logit-adjusted cross-entropy with `lambda=1.0`; evaluation retains the natural
class distribution. Primary metrics are macro-F1 and balanced accuracy, with
accuracy, weighted-F1, per-class precision/recall/F1, confusion matrix,
predicted-class counts, and one-vs-rest ROC-AUC/PR-AUC as required supporting
outputs. Always-`STABLE` and repeated training-prior scores are non-trained
references.

## 7. Canonical framework and model lifecycle

Phase 5 uses only the canonical five representation branches:

| Branch | Phase 5 role |
|---|---|
| Statistical | AR/GARCH deterministic features |
| Transformed | FFT/Haar deterministic features |
| VAE | Canonical generative encoder |
| Contrastive | Canonical CNN NT-Xent encoder |
| BYOL | Canonical CNN BYOL encoder |

Primary fusion is concatenation with the lightweight task head. Encoder
variants, gated fusion, branch ablations, duplicate-width controls, temporal
transfer, and fixed-first-walk encoder reuse are deferred to Phase 6.

Walk 1 and Walk 2 use the same declared architecture, training recipe, seeds,
and fixed budgets, but they have independently fitted weights. Each walk owns
its preprocessing state, neural-encoder checkpoints, feature store,
downstream heads, and baseline weights. Phase 1--3 checkpoints are historical
evidence and are not Phase 5 model inputs.

## 8. Baseline and comparison rules

All strict comparators must consume identical saved row identities, targets,
OHLCV information, activity eligibility, and walk boundaries. The initial
movement-regression comparison includes the exact-zero reference,
Raw-OHLCV MLP, raw sequential LSTM, and canonical five-branch framework. The
movement-classification comparison includes always-`STABLE`, training-prior
scores, Raw-OHLCV MLP, raw LSTM, adapted TA-MLP on the common TA-eligible row
intersection, and the canonical framework.

The exact learned baseline matrix, seeds, budgets, and optimization recipes
must be frozen in an implementation amendment before training. No validation
split, early stopping, or evaluation-driven checkpoint selection is used.

## 9. Deferred work

The following are outside the frozen Phase 5 core:

- encoder-backbone variants and additions;
- single-branch and leave-one-out ablations;
- gated aggregation and temporal-transfer ablations;
- alternative context lengths or classification thresholds;
- a redesigned volatility target and the corresponding Raw LSTM,
  GARCH--LSTM, GINN, and framework volatility matrix; and
- alpha-factor or trading-strategy evaluation.

Encoder variants and representation ablations move to Phase 6. Volatility may
be added only after its future, non-overlapping target and capacity are
separately specified and approved.

## 10. Phase 5 implementation gate

The data acquisition, approved pruning, bounded-fill artifacts, exploratory
statistics, and plots already exist. Training remains blocked only until the
following experiment-specific work is implemented and validated under
`scripts_v2/`:

1. build the two walk-specific sequence and identity manifests;
2. enforce the context, activity, observed-endpoint, target-maturity, and
   boundary rules above;
3. build shared two-hour regression and classification label bundles;
4. validate identical framework/baseline rows and raw-volume mask invariants;
5. implement independent walk-specific preprocessing and model lifecycle;
6. freeze the learned model/baseline matrix, seeds, budgets, and launch order;
7. complete CPU smoke tests and prediction replay checks; and
8. only then launch the Phase 5 training runs.

Retrospective universe selection and `quarantine_available_at` replay are not
Phase 5 blockers under the accepted assumptions in Section 3.
