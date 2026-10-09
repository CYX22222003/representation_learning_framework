# Phase 6.9 Task-Specific Competitiveness Demonstration

**Date:** 2026-10-09
**Active classification amendment:**
`2026-10-09-phase-6-9-sgn-classification-amendment.md` replaces the
unimplemented Monotone-VI classification leg with an independently authored
SGN adaptation (`SGN-C`). Comparator role and exact-row evaluation are
approved and complete: both 50-epoch trajectories and all six snapshots pass
same-backend replay, with matched reporting under
`experiments/phase6_9/sgn_classification/reports/seed0/`.
Monotone-VI-Frozen remains unchanged in optional Phase 6.8.

The SGN dossier now exists under `docs/baselines/SGN/`. Its paper reading and
pinned static source/settings/licence audit are complete. The owner approved
all seven decisions on 2026-10-09. The independent model/lifecycle, 17 CPU
tests, both train-only initializers, two-run matrix, and real-row CUDA
batch-256 admission, both 50-epoch trajectories, all six snapshots, same-
backend replay, and matched reporting are complete. See
`2026-10-10-phase-6-9-sgn-execution.md`.

**Active price amendment:**
`2026-10-09-phase-6-9-xlstm-endpoint-amendment.md` supersedes the former
full-path price contract. The primary method is now **XM-C8**, direct
`close[t+8h]` with endpoint-only MSE on every original price row. The
completed **XM-MV8** runs below are preserved historical/contextual evidence;
they do not complete XM-C8, and the former intersection-control reruns are
no longer required. Both fresh XM-C8 walks now complete 50 epochs; all six
snapshots pass independent replay and the endpoint-matched comparison is
recorded under `experiments/phase6_9/xlstm_mixer_endpoint/reports/seed0/`.
XM-C8 improves MAE/RMSE versus H0 and Raw LSTM in both walks. Persistence
remains better on MAE in both walks and RMSE in Walk 2. SGN classification is
also complete.

**Status:** Approved contract; xLSTM-Mixer paper/source audit, all fourteen
owner decisions, model, data builder, and local vanilla-GPU training launcher
are implemented. Both common-path bundles pass source replay. The 26 focused
tests include real vanilla-GPU interrupted/resumed fixture training and
same-backend replay. Both real-data seed-0 xLSTM-Mixer trajectories now
complete 50 epochs, and all six 5/15/50 snapshots pass standalone same-backend
checkpoint/prediction/metric replay. The XM-only diagnostic report is under
`experiments/phase6_9/xlstm_mixer/reports/seed0/`; matched H0-D0/Raw-LSTM
intersection reruns were originally required but are now superseded by the
endpoint amendment. Fresh XM-C8 training and matched reporting are now complete;
see `2026-10-09-phase-6-9-xlstm-endpoint-execution.md` for principal results and
the reporting-only schema recovery without training-fingerprint migration.
The real-data local vanilla-GPU admission and manifest-only launch pipeline
pass, with batch 512 and a one-row remainder under the 6 GiB admission ceiling.
The completed Phase 6.5B volatility artifacts are reused
rather than retrained.

**Closure update (2026-10-10):** all three task-specific legs, their replay,
matched task reports, and the final task-separated synthesis are complete. The
integrated report is
`experiments/phase6_9/reports/task_specific_competitiveness_seed0/summary.md`.
Phase 6.9 is closed for its frozen seed-0 scope.

**Primary reference:**
`2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`

**Optional parallel plan:**
`2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md`

**xLSTM-Mixer audit dossier:** `docs/baselines/xLSTM-Mixer/`

## 1. Decision and purpose

### Historical full-path execution evidence (2026-10-09)

The owner-requested local launch trained both `XM-MV8` walks with the frozen
seed-0, batch-512, Adam-1e-4, 50-epoch recipe. Epoch 50 remained predeclared;
all 5/15/50 snapshots are retained. Headline MAE/RMSE are
`0.003054837/0.010852719` (Walk 1) and `0.004780288/0.021858833` (Walk 2).
Implied-movement Spearman is `0.352519/0.257461`. Against identical-row
persistence, MAE worsens in both walks; RMSE improves only in Walk 1. These
are mixed XM-only diagnostics, not a completed learned-baseline comparison.
Full eight-hour five-channel supervision and unconstrained OHLC/volume
diagnostics are disclosed in the run summaries and
`experiments/phase6_9/xlstm_mixer/reports/seed0/execution_record.md`.

Phase 6.9 adds a bounded task-specific comparison alongside the completed
recent representation-learning comparison. It asks whether one reusable
canonical representation remains competitive when each downstream task is
compared with a recent or domain-aligned method chosen for that task:

| Task | Phase 6.9 comparator | Comparison role |
|---|---|---|
| Two-hour movement classification | SGN-C classification adaptation (NeurIPS 2025) | Supervised classification-specific complete-system baseline |
| Eight-hour future price | xLSTM-Mixer XM-C8 endpoint adaptation (NeurIPS 2025) | Endpoint-matched complete-system baseline |
| Eight-hour future realised variance | Adapted GARCH--LSTM stack | Finance-oriented volatility complete-system baseline |

The research contribution remains the unified representation-learning
framework. Phase 6.9 does not claim that the three comparators form a uniform
architecture family or that their metrics can be averaged into one league
table. Instead, it adds financial and task-level context to the direct
representation evidence already produced by Phase 6.7.

Phase 6.8 remains a frozen but optional, unimplemented representation-baseline
extension. It is not cancelled, rewritten, or a prerequisite for Phase 6.9.
Phase 7A and Phase 6.6's deferred raw-fusion/decoder work remain separately
planned. Phase 6.6 is no longer an active xLSTM-Mixer authority.

## 2. Research questions

1. Can the same canonical `H0` representation support classification, future
   price, and future realised variance while remaining competitive with a
   source-aligned comparator for each individual task?
2. Which differences are consistent across both calendar walks and the
   principal task metrics?
3. What specialization, supervision, parameter, runtime, and implementation
   costs accompany each task-specific result?

The phase distinguishes two claims:

- **transferability:** one target-free `H0` representation is reused across
  all three tasks within each walk; and
- **task competitiveness:** each task is compared separately with its named
  task-oriented baseline.

Neither claim requires `H0` to beat every specialized method. A specialized
winner may instead quantify the cost of using one reusable representation.

## 3. Shared data and fairness contract

Reuse the accepted one-hour global-calendar walks:

| Walk | Permitted training interval | Evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

Every fitted object remains walk-specific. No later-walk data may update or
select an earlier-walk model. Phase 6.9 retains:

- the saved 64-by-5 decision-time OHLCV contexts;
- the canonical epoch-50 `H0` representations;
- the existing task definitions, train/evaluation identities, references,
  losses, and metric implementations; and
- the train/test-only rule: no evaluation-driven architecture choice, early
  stopping, restart selection, or best-on-test checkpoint selection.

Model-specific row deletion is prohibited. XM-C8 reuses every original h8
price row and its sole endpoint target. Additional future-path availability
is not an active eligibility rule; the earlier XM-MV8 intersection is
historical only.

## 4. Classification: independent SGN adaptation

### 4.1 Role

The owner-approved
`2026-10-09-phase-6-9-sgn-classification-amendment.md` selects SGN, an
explicit supervised multivariate time-series classifier, instead of the
previously planned Monotone-VI representation-plus-probe comparison.
Write the SGN model and lifecycle independently, using the paper and pinned
official source/settings as references. This is a project adaptation, not a
claim of exact paper reproduction or finance-specific source validation.

The mandatory primary row is:

```text
SGN-C: historical raw OHLCV -> supervised SGN backbone -> native 3-class head
```

Compare with immutable `H0-D0` and Raw LSTM on every original two-hour
`DOWN/STABLE/UP`, `tau=0.001` row for both walks. This is a complete-system
competitiveness comparison, not a shared-head representation control.
The optional Phase 6.8 Monotone-VI-Frozen plan remains unchanged.

### 4.2 Source/adaptation and causal initialization gate

Before implementation, approve a separate SGN specification and complete
its pinned paper/source/settings/licence audit. Preserve the defining
grouping, multi-scale mixing, shifted-window, merging, and native-head
mechanisms, with explicit decisions for paper/source discrepancies.

1. Derive similarity/BDC, K-means, any fitted FFT/period statistics, and
   preprocessing from the corresponding classification training population only.
2. Freeze the data-derived initialization and full training recipe before
   evaluation; never reuse upstream sensor matrices or test-derived metadata.
3. Train independent supervised weights per walk and predict every unchanged
   evaluation row without any shared-state refitting on evaluation data.

Resolve group count/embedding semantics, period selection, 64-step-safe
depth/merging, temperature state, regularizer, optimizer/batch, and replay
behavior in the implementation discussion. This plan does not silently
approve numerical architecture or optimizer choices, copying unlicensed
source, or dropping inconvenient rows.

### 4.3 Fixed lifecycle and supervision boundary

Use two fresh seed-0 50-epoch supervised trajectories, retain 5/15/50, and
keep epoch 50 primary. Use the existing training-prior logit-adjusted task
loss; SGN's additional grouping regularizer must be separately frozen and
disclosed. No validation early stopping or best-on-test selection is allowed.
The paper's dataset-specific configurations are references, not automatic
overrides of this lifecycle. The former MVI common-probe and source-KNN/SVM
sensitivities are superseded, not completed. No frozen-SGN probe or extra
task/ablation matrix is commissioned.

## 5. Future price: xLSTM-Mixer

### 5.1 Phase 6.9 ownership

Phase 6.9 is the sole technical, source-alignment, endpoint-availability,
architecture, training, artifact, and reporting authority for xLSTM-Mixer.
The prior Phase 6.6B listing is retained only as a superseded historical
reference. It cannot launch or own a second run if Phase 6.6 later resumes.

Active XM-C8 artifacts live under
`experiments/phase6_9/xlstm_mixer_endpoint/`. Historical XM-MV8 artifacts
remain unchanged under `experiments/phase6_9/xlstm_mixer/`.

### 5.2 Inherited experiment contract

Use the owner-approved **XM-C8** amendment:

- input: the same 64-hour five-channel OHLCV context;
- supervision/output: one scalar `close[t+8h]`, raw-probability MSE;
- model: audited xLSTM-Mixer core with a direct endpoint adapter, not the
  source paper's full-path objective;
- fairness: every original h8 training/evaluation row, with unchanged
  identities, ordering, historical contexts, and endpoint targets; and
- controls: reuse original H0-D0/Raw-LSTM runs only after source, identity,
  target, recipe, and prediction-integrity checks. No intersection reruns.

For paper/source discrepancies, the pinned official implementation is the
behavioral authority. The primary model uses the released latent-feature-axis
reversal and non-affine RevIN. The official source
reverses latent feature coordinates rather than variate-token order, disables
RevIN affine parameters, and tunes zero to four initial tokens. The final
paper/repository revision and runtime/package details also do not align
exactly. Phase 6.9 owns the decisions recorded in
`docs/baselines/xLSTM-Mixer/upstream_clarification_request.md`; it may not
silently preserve the older provisional diagram.

The approved project adaptation uses the compact `D=128`, one-block,
eight-head full model, kernel 0, dropout 0.1, packing 1, and the NLinear
backbone with exactly one learned initial token initialized from
`Normal(0,0.01)`. Training uses the project lifecycle:
float32, seed 0, Adam at `1e-4`, no weight decay, 50 epochs, snapshots
5/15/50, and epoch 50 primary. Physical batch 512 is attempted first and may
be reduced only by a pre-training resource decision shared by both
walks. XM-C8 uses endpoint-only mean MSE rather than full-path L1. Its shared
time/up/down projections are `64 -> 1`, `1 -> 128`, and `256 -> 1`.
Only the mixed close token is decoded and inverse-normalized; other variates
remain historical inputs, not supervised future outputs. Inverse RevIN is
unclipped; report out-of-range forecasts. Clip norm 1.0 is retained and
disclosed alongside the sigmoid/no-clipping controls.

No second xLSTM-specific channel scaler is fitted. OHLC stays in `[0,1]`, and
the accepted Phase 5 walk-training-only volume transform is reused for context
only; there are no future volume labels.

**Owner-directed local-runtime amendment (2026-10-09):** Lumid is no longer
an execution prerequisite. The primary local path uses the isolated
`.venv-xlstm-mixer/`, Python 3.10.12, PyTorch `2.12.1+cu126`, and pinned
`xlstm==1.0.3` vanilla sLSTM on CUDA tensors. The disclosed lazy CUDA-loader
patch only removes import-time toolkit discovery; it does not change sLSTM
math. This path needs neither nvcc nor Docker. The compiled `cuda` backend
remains an explicitly selected alternative requiring its own admission.
Backend, dependency source hash, runtime versions, code/data identities,
batch-512 and one-row backward/update/replay probes, and resource limits are
recorded and guarded. Architecture, data, optimizer, seed, budget, supervision,
comparison requirements, and epoch-50 selection rule were unchanged by the
runtime amendment. The later endpoint amendment changes target/row/loss
scope and requires fresh XM-C8 admission and weights.

The earlier read-only check documented the now-historical XM-MV8 intersection:

| Walk | Split | Existing price rows | Fully observed common rows |
|---:|---|---:|---:|
| 1 | train | 36,773 | 32,470 |
| 1 | evaluation | 29,834 | 27,786 |
| 2 | train | 56,652 | 53,112 |
| 2 | evaluation | 13,506 | 12,115 |

XM-C8 instead preserves all **existing price rows** in the third column;
there is no future-path builder or additional observation filter. There is no validation split or
cross-backend CPU/CUDA numerical-parity gate. Lightweight same-CUDA-backend
checkpoint reload, fixed-probe, prediction, identity, and metric replay remain
required artifact-integrity checks.

Historical XM-MV8 tables must disclose additional channel/intermediate-horizon
supervision and must not be relabelled as XM-C8. XM-C8 is endpoint- and
row-matched but still a complete-system comparison, not a causal test of
representation quality or sLSTM versus ordinary LSTM.

## 6. Realised variance: completed GARCH--LSTM

Phase 6.9 reuses the replay-valid Phase 6.5B strict H=8 GARCH--LSTM stacks. No
new volatility model training is required. The final task table joins, by
exact walk and row identity:

- canonical `H0-D0`;
- Raw LSTM;
- adapted GARCH--LSTM stacking; and
- the established zero, train-location, and historical-volatility-persistence
  references.

The GARCH--LSTM result remains a complete-system hybrid comparison. Its
chronological expanding OOF base predictions train the ElasticNet meta-model;
they are not a validation split. Existing evidence—marginal epoch-50 MSE
improvement over Raw LSTM but worse MAE and Spearman in both walks—must be
retained rather than reframed as a broad win.

## 7. Planned inventory

The new mandatory execution inventory is bounded:

```text
SGN-C:       2 fresh walk-specific supervised classification trajectories
             with snapshots at 5/15/50; no representation-store stage
XM-C8:       2 fresh endpoint-only price trajectories
GARCH-LSTM:  0 new trajectories; reuse 2 completed Phase 6.5B stacks
```

Original H0-D0 and Raw LSTM results are reused after exact-source/identity
verification; former intersection reruns are superseded, not completed.
The former Monotone-VI fits, stores, and source-classifier sensitivity are
superseded in Phase 6.9, not commissioned alongside SGN.

This is not a homogeneous model-by-task matrix. SGN is not forced onto
price or volatility, xLSTM-Mixer is not forced onto classification or
volatility, and GARCH--LSTM is not forced onto classification or price.

## 8. Evaluation and reporting

Report each task separately and each walk before any pooled summary:

- **classification:** macro-F1 is primary; also accuracy, balanced accuracy,
  per-class precision/recall/F1, confusion matrix, predicted-class counts,
  and collapse diagnostics;
- **future price:** MAE and RMSE are primary; also MSE, Pearson/Spearman,
  persistence skill, implied-movement correlation/sign agreement, global and
  timestamp-level Rank IC, subgroup diagnostics, and endpoint replay; and
- **realised variance:** MAE and RMSE are primary; also MSE,
  Pearson/Spearman, persistence skill, contract-macro, and tail/subgroup
  diagnostics.

Report parameters or solver-state size, fitting/training time, inference time,
peak memory, implementation origin, source deviations, and additional
supervision. Do not average metrics or ranks across the three tasks. Preserve
negative and mixed results.

The final interpretation may state that one reusable representation is
competitive on named tasks and walks, or that specialization retains an
advantage on a named task. It may not claim universal SOTA, profitable alpha,
architecture causality, or multi-seed robustness.

## 9. Code, artifacts, and ordered gates

Proposed SGN code and Phase 6.9 integration locations, subject to the separate
implementation specification, are:

```text
docs/baselines/SGN/
src/baselines/sgn/
scripts_v8/
tests/baselines/sgn/
experiments/phase6_9/
  sgn_classification/
    feasibility/
    manifests/
    downstream/classification_h2/walk{1,2}/sgn_c/seed0/
    reports/seed0/
  xlstm_mixer/
    feasibility/
    data/absolute_price_h8_multivariate/
    downstream/absolute_price_h8/walk{1,2}/xm_mv8/seed0/
    matched_controls/absolute_price_h8/
    diagnostics/resources/
  xlstm_mixer_endpoint/
    feasibility/
    manifests/
    downstream/absolute_price_h8/walk{1,2}/xm_c8/seed0/
    reports/seed0/
  reports/task_specific_competitiveness_seed0/
```

xLSTM-Mixer implementation and model artifacts use the Phase 6.9 paths below.
GARCH--LSTM remains immutable under `experiments/phase6_5/`.

Ordered gates are:

1. Replay the two walk bundles, the three exact task populations, canonical
   `H0`, Raw LSTM references, and completed GARCH--LSTM artifacts.
2. Complete the SGN paper/source/settings/licence audit and discuss/approve
   the independent implementation specification before model coding.
3. Implement and CPU-test the model/lifecycle; verify all original
   classification rows and controls, freeze the two-run manifest, and admit
   the declared GPU recipe with bounded resource checks.
4. Only after explicit execution approval, train both SGN-C walks and
   independently replay all six classification snapshots and matched reports.
5. Audit every original h8 price row and the existing H0-D0/Raw-LSTM controls;
   freeze the XM-C8 endpoint recipe and fresh selected-backend CUDA admission.
6. Train fresh XM-C8 once per walk, preserve scalar endpoint predictions,
   and replay checkpoints, exact identities, targets, histories, and metrics.
7. Revalidate, but do not retrain, the two Phase 6.5B GARCH--LSTM stacks and
   join their exact-row results with `H0` and Raw LSTM.
8. Generate one task-separated Phase 6.9 report with provenance links and
   claim boundaries.

Launchers remain audit/manifest-only by default and require an explicit
execution flag for fitting or training.

## 10. Exit conditions

Phase 6.9 is complete only when:

- the SGN paper/source/settings/licence audit and independently authored
  implementation specification are approved and causal initialization covers
  every original classification row without changing the task;
- both `SGN-C` trajectories, all 5/15/50 snapshots, and original classification
  controls pass checkpoint/state, preprocessing, identity, prediction, and
  metric replay, with a matched classification report;
- the Phase 6.9 source/hardware/endpoint gate is frozen and both `XM-C8`
  walk trajectories plus verified original controls pass same-runtime checkpoint,
  identity, prediction, and metric replay;
- the completed GARCH--LSTM artifacts are revalidated and joined without
  changing their prior result interpretation;
- every comparison discloses its supervision and comparison level; and
- the final report preserves the two-walk, seed-0, non-trading,
  non-universal claim boundary.

SGN classification and all three Phase 6.9 technical legs are complete. The
final task-separated synthesis is generated, so Phase 6.9 is closed for its
frozen seed-0 scope. Any further comparator replacement or expanded matrix
requires a separate owner decision.
