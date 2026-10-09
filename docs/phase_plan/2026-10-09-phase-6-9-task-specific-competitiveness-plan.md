# Phase 6.9 Task-Specific Competitiveness Demonstration

**Date:** 2026-10-09
**Status:** Approved planning contract; xLSTM-Mixer paper/source audit and all
fourteen owner decisions are complete. Implementation has not started; its
common-intersection target artifact and Lumid Sandbox CUDA admission remain
required before training. The completed Phase 6.5B volatility artifacts are
reused rather than retrained.

**Primary reference:**
`2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`

**Optional parallel plan:**
`2026-10-06-phase-6-8-recent-conference-representation-baselines-plan.md`

**xLSTM-Mixer audit dossier:** `docs/baselines/xLSTM-Mixer/`

## 1. Decision and purpose

Phase 6.9 adds a bounded task-specific comparison alongside the completed
recent representation-learning comparison. It asks whether one reusable
canonical representation remains competitive when each downstream task is
compared with a recent or domain-aligned method chosen for that task:

| Task | Phase 6.9 comparator | Comparison role |
|---|---|---|
| Two-hour movement classification | Monotone-VI classification adaptation (ICLR 2025) | Recent classification-oriented representation baseline |
| Eight-hour future price | xLSTM-Mixer (NeurIPS 2025) | Recent multivariate forecasting complete-system baseline |
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

Model-specific row deletion is prohibited. A source method requiring an
additional observed target path may use only a metadata-defined common
intersection frozen before training, with required comparator reruns on the
same training and evaluation identities.

## 4. Classification: Monotone-VI adaptation

### 4.1 Role

*Nonlinear Sequence Data Embedding by Monotone Variational Inequality* is an
unsupervised sequence-representation method whose principal real-time-series
experiments evaluate classification and clustering. Phase 6.9 therefore uses
it as a classification baseline. It is not described as a finance-specific
classifier, an end-to-end forecasting model, or evidence for the price and
volatility tasks.

The mandatory primary row is:

```text
MVI-C-D0: train-only Monotone-VI representation -> established simple
          classification probe
```

It is compared directly with immutable `H0-D0` on the exact two-hour
`DOWN/STABLE/UP`, `tau=0.001` rows for each walk. This keeps the supervised
probe fixed and attributes the primary difference to the representations.

### 4.2 Inductive admission gate

Monotone-VI learns shared low-rank structure across a collection of
sequences. Before implementation, its paper and source dossier must establish
an inductive evaluation procedure:

1. fit every shared basis, operator, normalizer, solver choice, and other
   learned state using the permitted walk-training population only;
2. freeze that state before any evaluation sequence is embedded; and
3. embed each evaluation sequence without jointly refitting on training plus
   evaluation sequences.

If no defensible inductive extension exists, Monotone-VI is not admitted to
the headline Phase 6.9 comparison. A separately labelled transductive
sensitivity may be proposed only through a dated amendment and cannot be
reported beside `H0` as a strict held-out result.

The dossier also freezes input mapping, finite-difference handling, lookback,
link function, regularization, solver iterations/tolerance, representation
width, seed semantics, and failure policy without reading Polymarket
evaluation metrics.

### 4.3 Optional source-style classifier sensitivity

The source paper evaluates embeddings using KNN and RBF-SVM classifiers. A
source-style sensitivity may be added only if its classifier and
hyperparameters are fixed before evaluation without introducing the project's
prohibited best-on-test selection. If commissioned, the identical classifier
recipe must also be applied to `H0`:

```text
MVI-C-SRC versus H0-SRC
```

This sensitivity is reported separately from `MVI-C-D0 versus H0-D0`. It may
not be used to attribute an improvement to the representation when the
classifier or tuning budget differs.

## 5. Future price: xLSTM-Mixer

### 5.1 Phase 6.9 ownership

Phase 6.9 is the sole technical, source-alignment, full-path-availability,
architecture, training, artifact, and reporting authority for xLSTM-Mixer.
The prior Phase 6.6B listing is retained only as a superseded historical
reference. It cannot launch or own a second run if Phase 6.6 later resumes.

Canonical model artifacts live under
`experiments/phase6_9/xlstm_mixer/`.

### 5.2 Inherited experiment contract

Use the Phase 6.9 `XM-MV8` contract:

- input: the same 64-hour five-channel OHLCV context;
- supervision: the complete observed next-eight-bar, five-channel path;
- model: source-aligned xLSTM-Mixer with sLSTM blocks;
- project endpoint: extract only `close[t+8]` for the established future-price
  comparison; and
- fairness: freeze full-path availability from metadata before training and
  rerun `H0-D0` and Raw LSTM on the same common intersection.

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
be reduced only by a pre-training Lumid resource decision shared by both
walks. The method-specific unweighted full-path L1 objective is retained.

No second xLSTM-specific channel scaler is fitted. OHLC stays in `[0,1]`, and
the accepted Phase 5 walk-training-only volume transform is reused for context
and future-target volume. The experiment runs in the persistent Lumid Sandbox;
the Docker/devcontainer environment is a development/workflow reference.

The read-only identity check confirms that a common intersection is required:

| Walk | Split | Existing price rows | Fully observed common rows |
|---:|---|---:|---:|
| 1 | train | 36,773 | 32,470 |
| 1 | evaluation | 29,834 | 27,786 |
| 2 | train | 56,652 | 53,112 |
| 2 | evaluation | 13,506 | 12,115 |

The data builder must freeze and assert these identities while materializing
`[8,5]` targets. This bounded metadata join replaces a broader exploratory
availability study, but it cannot be skipped. There is no validation split or
cross-backend CPU/CUDA numerical-parity gate. Lightweight same-CUDA-backend
checkpoint reload, fixed-probe, prediction, identity, and metric replay remain
required artifact-integrity checks.

Every table must disclose that `XM-MV8` receives additional channel and
intermediate-horizon supervision. Its result is a complete-system comparison,
not a target-matched causal test of representation quality or sLSTM versus
ordinary LSTM.

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
Monotone-VI: 2 walk-specific train-only fits
Monotone-VI: 2 complete classification representation stores
MVI-C-D0:    2 classification probe trajectories, snapshots at 5/15/50
XM-MV8:      2 source-aligned price trajectories
GARCH-LSTM:  0 new trajectories; reuse 2 completed Phase 6.5B stacks
```

The observed-path identity check already establishes a reduced population, so
the predeclared matched `H0-D0` and Raw-LSTM reruns are mandatory. If the optional source-style
classification sensitivity is commissioned, both `MVI-C-SRC` and `H0-SRC`
are added together.

This is not a homogeneous model-by-task matrix. Monotone-VI is not forced onto
price or volatility, xLSTM-Mixer is not forced onto classification or
volatility, and GARCH--LSTM is not forced onto classification or price.

## 8. Evaluation and reporting

Report each task separately and each walk before any pooled summary:

- **classification:** macro-F1 is primary; also accuracy, balanced accuracy,
  per-class precision/recall/F1, confusion matrix, predicted-class counts,
  and collapse diagnostics;
- **future price:** MAE and RMSE are primary; also MSE, Pearson/Spearman,
  persistence skill, implied-movement correlation/sign agreement, global and
  timestamp-level Rank IC, subgroup diagnostics, and full-path replay; and
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

New Monotone-VI code and Phase 6.9 integration artifacts belong under:

```text
src/baselines/monotone_vi/
scripts_v8/
tests/baselines/monotone_vi/
experiments/phase6_9/
  feasibility/monotone_vi/
  manifests/
  representation_fitting/monotone_vi_classification/
  features/classification/walk{1,2}/
  downstream/classification_h2/walk{1,2}/mvi_c_d0/seed0/
  xlstm_mixer/
    feasibility/
    data/absolute_price_h8_multivariate/
    downstream/absolute_price_h8/walk{1,2}/xm_mv8/seed0/
    matched_controls/absolute_price_h8/
    diagnostics/resources/
  reports/task_specific_competitiveness_seed0/
```

xLSTM-Mixer implementation and model artifacts use the Phase 6.9 paths below.
GARCH--LSTM remains immutable under `experiments/phase6_5/`.

Ordered gates are:

1. Replay the two walk bundles, the three exact task populations, canonical
   `H0`, Raw LSTM references, and completed GARCH--LSTM artifacts.
2. Complete the Monotone-VI paper/source/licence/adaptation dossier and prove
   inductive train-only fitting before coding or evaluation.
3. Freeze the two Monotone-VI fits, two stores, two common-probe runs, and any
   optional matched source-classifier sensitivity.
4. Implement, CPU-test, resource-audit, fit, extract, probe, and replay the
   complete Monotone-VI classification scope.
5. Build the deterministic fully observed future-path intersection and
   targets for the owner-approved xLSTM-Mixer recipe, and
   freeze the exact recipe and Lumid Sandbox runtime admission.
6. Train and replay `XM-MV8` once per walk plus the required matched controls,
   preserving complete path predictions.
7. Revalidate, but do not retrain, the two Phase 6.5B GARCH--LSTM stacks and
   join their exact-row results with `H0` and Raw LSTM.
8. Generate one task-separated Phase 6.9 report with provenance links and
   claim boundaries.

Launchers remain audit/manifest-only by default and require an explicit
execution flag for fitting or training.

## 10. Exit conditions

Phase 6.9 is complete only when:

- the Monotone-VI dossier establishes an inductive train-only procedure and
  both fitted representations cover every frozen classification row;
- both `MVI-C-D0` trajectories and all 5/15/50 snapshots pass representation,
  scaler, identity, prediction, and metric replay;
- the Phase 6.9 source/hardware/full-path gate is frozen and both `XM-MV8`
  walk trajectories plus the matched controls pass same-runtime checkpoint,
  identity, prediction, and metric replay;
- the completed GARCH--LSTM artifacts are revalidated and joined without
  changing their prior result interpretation;
- every comparison discloses its supervision and comparison level; and
- the final report preserves the two-walk, seed-0, non-trading,
  non-universal claim boundary.

Failure of Monotone-VI's inductive admission gate is a documented method
rejection, not permission to fit on evaluation sequences. A replacement
classification baseline would require a dated amendment.
