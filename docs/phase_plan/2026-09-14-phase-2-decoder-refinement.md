# Phase 2 Decoder Refinement Implementation Specification

Date: 2026-09-14

## Purpose and status

This document freezes the implementation contract for Part 1 of the Phase 2
experiment plan. It converts the approved `D0`--`D4` decoder matrix into a
concrete code, data, execution, and reporting plan. The Stage-1 data contracts,
D0--D4 models, trainer, bootstrapper, reporter, and tests are implemented. The
30-run CUDA matrix has been frozen but not executed.

The experiment asks whether the fixed five-branch Phase-1 representation is
limited by static head capacity, branch fusion, or the lack of temporal
modelling across consecutive embeddings. It does not retrain or replace the
Phase-1 VAE, contrastive, or BYOL encoders.

## Frozen experiment contract

| Setting | Frozen value |
|---|---|
| Source data | `data/processed/market_4h_seq64_top50.npz` |
| Frozen features | `data/features/features_4h_seq64_top50_phase1.npz` |
| Branch order | `statistical`, `transformed`, `vae`, `contrastive`, `byol` |
| Concat dimension | 445 |
| Temporal context `K` | 8 consecutive representation rows |
| Seeds | `0,1,2` |
| Epoch snapshots | `15,50,100` from one uninterrupted trajectory |
| Batch size | 512 |
| Optimizer | Adam, learning rate `1e-4`, weight decay `0` |
| Training control | fixed budgets; no validation split or early stopping |
| Feature scaling | per-branch mean/std fitted on the full training feature split; clip to `[-10,10]` |
| Primary tasks | price prediction and volatility prediction |
| Later supported task | Phase-2 probability-movement classification using fixed `P2` |

`K=8` gives 32 hours of representation endpoints at the 4-hour timeframe,
while every endpoint representation already summarizes a 64-row input window.
It is large enough to test dynamics between embeddings without making the
temporal models unnecessarily large or discarding many task rows. It is a
single frozen value, not a test-tuned hyperparameter.

The current task-test partition has already informed Phase 2. Results from this
matrix are characterization evidence; a combined final design requires a fresh,
temporally later holdout for confirmation.

## Decoder matrix and exact architectures

All decoders receive standardized frozen branches. Static decoders are trained
on the same final target rows retained by the temporal decoders, so row-count
differences cannot explain a result.

| ID | Input | Predeclared architecture |
|---|---|---|
| `D0` | final 445-d concat row | Existing shallow task MLP with hidden widths `128,64` |
| `D1` | five named branches at final row | Each branch projected to 64; concatenated; projected to 128; two pre-norm residual MLP blocks with expansion 2 and dropout 0.1 |
| `D2` | five named branches at final row | Newly initialized task-specific `RepresentationAggregator(mode="gated", out_dim=128)` followed by a newly initialized copy of the existing shallow task-head architecture |
| `D3` | `[K,445]` | Linear input projection to 128, LayerNorm, input dropout 0.1, one-layer LSTM with hidden size 128, final valid hidden state, task readout |
| `D4` | `[K,445]` | Linear projection to 96, sinusoidal positions, two pre-norm Transformer encoder blocks, four heads, feed-forward width 192, dropout 0.1, causal mask, final-token readout |

Here, "existing" refers to repository class definitions and architecture, not
to pretrained Phase-1 weights. Phase 1 used concat fusion, whose aggregator has
no learnable parameters, and did not train a gated aggregator. Each D2
task/seed trajectory therefore creates a fresh gate and task head and optimizes
them jointly from scratch on that task's training rows. Only the five saved
branch features are frozen. D2 weights are neither shared between tasks or
seeds nor supplied to D3/D4; D2 is a parallel static fusion control, not a
pretraining stage for the temporal decoders.

The `D1`, `D3`, and `D4` implementations should have approximately 0.20M
trainable parameters for scalar regression and must remain within 10% of one
another. Exact counts are recorded and asserted by tests. `D0` is intentionally
smaller, and `D2` reuses the repository's gated class implementation with fresh
task-specific weights; neither should be padded with unused parameters merely
to match the other models.

The shared task readout is LayerNorm, a linear reduction to half the decoder
width, GELU, dropout 0.1, and the task output layer. `D0` and `D2` retain their
existing task heads rather than using this readout, because their purpose is to
control against current repository behavior.

Task outputs are fixed as follows:

- Price: one unconstrained scalar, trained with MSE. A sigmoid/bounded-price or
  residual-change formulation would be a separate experiment and is excluded.
- Volatility: one scalar followed by Softplus for every `D0`--`D4` Phase-2 run,
  trained with MSE. The historical Phase-1 linear-output run is reported as a
  contextual reference, not inserted into the matched matrix.
- Classification: three unadjusted logits. If run after Part 3, use natural
  sampling with the fixed `P2` train-prior logit-adjusted cross-entropy; report
  unadjusted-logit softmax scores at inference.

`D1` and `D2` are static capacity/fusion controls. Only `D3` and `D4` support a
temporal-decoder claim.

## Temporal row-map and target contracts

The Phase-1 feature store records split sizes but not contract identity or
window starts. Temporal decoding therefore requires a new immutable row-map
artifact before model training:

```text
data/features/phase2/
  temporal_index_4h_seq64_top50_k8.npz
  temporal_index_4h_seq64_top50_k8.npz.manifest.json
```

The preparation script reconstructs the top-50 contracts from raw Feather
files exactly as sequence preparation did, verifies every reconstructed split
against the processed NPZ, and emits for each split:

- `context_row_indices`: shape `[M,K]`, indices into that processed/feature split;
- `final_row_indices`: shape `[M]`, equal to the last column above;
- `contract_ids`, `final_window_starts`, and `final_timestamps_ns`;
- optional context window starts/timestamps for full replay diagnostics; and
- identity hashes, source checksums, excluded-row counts, and per-contract counts.

A valid context has one contract ID, strictly increasing row indices, window
starts increasing by exactly one, and strictly increasing timestamps wherever
timestamps are available. Contexts never cross train/test or contract
boundaries. Missing timestamps may be represented by `-1` and are diagnosed in
the manifest; they do not relax the row/window continuity requirement. The
implementation stores only indices and gathers branch arrays lazily, rather
than materializing the roughly 1.5 GB training tensor `[N,8,445]`.

Target eligibility is joined to `final_row_indices`:

- Price uses a new split-safe, contract-aware label bundle at horizon 1. This is
  required because the current generic price helper shifts a globally merged
  split and does not preserve contract identities at joins.
- Volatility reuses
  `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz` unchanged.
- Classification reuses the Phase-2 `h=2`, `tau=0.005` movement bundle. The
  decoder-only comparison may use its full label-eligible rows because no TA
  model is involved; it must not be mixed into the Part-3 C1/C2/C5 report.

With the current 50-contract artifacts and `K=8`, the expected matched row
counts are:

| Task | Train | Test |
|---|---:|---:|
| Price, horizon 1 | 109,441 | 27,100 |
| Volatility, horizon 1 | 109,441 | 27,100 |
| Movement classification, horizon 2 | 109,391 | 27,050 |

These counts are acceptance checks, not hard-coded loader logic. They must be
recomputed from manifests and fail closed if source data changes.

## Implemented code layout

The implementation uses a self-contained package so the stable Phase-1 runner
remains unchanged:

```text
src/tasks/phase2_decoders/
  __init__.py
  data.py          # row-map validation, target joins, lazy datasets, scaling
  models.py        # D0-D4 and task output adapters
  runner.py        # fixed-budget training, snapshot evaluation, replay
  reporting.py     # aggregation, paired uncertainty, tables and plots
  README.md
```

The project-root entry points are:

```text
scripts/prepare_phase2_temporal_index.py
scripts/prepare_price_labels.py
scripts/train_phase2_decoder.py
scripts/bootstrap_phase2_decoders.py
scripts/report_phase2_decoders.py
```

Responsibilities:

1. `prepare_phase2_temporal_index.py` builds or verifies the immutable `K=8`
   row map and its provenance manifest.
2. `prepare_price_labels.py` builds or verifies the split-safe price bundle,
   including labels, row indices, contract IDs, starts, timestamps, and hashes.
3. `train_phase2_decoder.py` runs exactly one task/decoder/seed trajectory,
   writes all checkpoints before performing task-test inference, then evaluates
   all snapshots and performs artifact replay. Test arrays may be loaded for
   alignment checks but are never used by the training loop.
4. `bootstrap_phase2_decoders.py` freezes the matrix and writes `commands.sh` by
   default. `--execute` runs it; complete runs are skipped, partial runs fail
   until explicitly inspected or overwritten.
5. `report_phase2_decoders.py` verifies identity hashes and replay status before
   producing per-task, multi-seed, resource, paired-difference, and per-contract
   reports.

Do not add decoder switches to `scripts/train_framework.py`; doing so would
increase the risk of changing Phase-1 behavior and would mix static and temporal
data contracts in one legacy entry point. Reuse small proven utilities by
extracting them into the new package only when tests preserve existing callers.

## Execution matrix

Stage 1 is the required decoder study:

```text
tasks    = price_prediction, volatility_prediction
decoders = D0, D1, D2, D3, D4
seeds    = 0, 1, 2
budgets  = 15, 50, 100 snapshots
```

This is 30 training trajectories and 90 evaluated snapshots. Each trajectory
trains uninterrupted to epoch 100. The launcher trains every declared snapshot
before task-test evaluation and does not select the best test epoch.

Stage 2 is optional until the Part-3 classification report is complete:

```text
task     = trend_classification
labels   = probability movement h=2, tau=0.005
protocol = P2 only
decoders = D0, D1, D2, D3, D4
seeds    = 0, 1, 2
budgets  = 15, 50, 100 snapshots
```

Stage 2 adds 15 trajectories and must use its own report under the decoder root.
No decoder outcome may be added to or ranked inside the Part-3 C1/C2/C5 matrix.

Commands:

```bash
.venv/bin/python3 scripts/prepare_phase2_temporal_index.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --features-npz data/features/features_4h_seq64_top50_phase1.npz \
  --context-length 8

.venv/bin/python3 scripts/prepare_price_labels.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/task_labels/price_prediction/price_4h_h1_seq64_top50.npz \
  --horizon 1

.venv/bin/python3 scripts/bootstrap_phase2_decoders.py

# Inspect matrix_manifest.json and commands.sh before execution.
.venv/bin/python3 scripts/bootstrap_phase2_decoders.py --execute

.venv/bin/python3 scripts/report_phase2_decoders.py \
  experiments/framework/phase2/decoder_refinement_1/4h_k8
```

## Artifacts and replay

Use only new Phase-2 paths:

```text
experiments/framework/phase2/decoder_refinement_1/4h_k8/
  matrix_manifest.json
  commands.sh
  <task>/<decoder>/seed<seed>/
    config.json
    dataset_manifest.json
    feature_standardizer.npz
    architecture.json
    timing.json
    training_diagnostics.json
    e<budget>/
      checkpoint.pth
      history.npz
      metrics.json
      predictions.npz
      per_contract_metrics.json
      replay.json
```

Prediction artifacts include predictions/targets plus final processed-row
index, contract ID, window start, and timestamp. D2 additionally saves gate
weights and branch order. Every run records source SHA-256 hashes, row identity
hashes, parameter count, wall-clock training time, inference time/throughput,
device, peak CUDA memory when applicable, and non-finite diagnostics.

## Evaluation and interpretation

Primary metrics:

- price: MAE and RMSE; also record MSE and Pearson correlation for consistency;
- volatility: MAE, RMSE, MSE, Pearson correlation, and negative-prediction
  fraction (which must be zero for the Softplus matrix); and
- classification: macro-F1, balanced accuracy, per-class metrics, confusion
  matrix, predicted-class counts, ROC-AUC, and PR-AUC.

For each task, seed, and budget, compare `D1`--`D4` with row-aligned `D0`.
Report absolute metrics and paired differences. For uncertainty, resample whole
contracts with replacement for 2,000 paired replicates using seed `20260914`
and report percentile 95% intervals. This preserves within-contract serial
dependence better than an independent row bootstrap. Across seeds, report mean
and sample standard deviation for every decoder and budget.

Interpretation is factorial:

- `D1 - D0`: added static nonlinear/branch-aware capacity;
- `D2 - D0`: learned branch weighting;
- `D3 - D1`: recurrence beyond a similarly sized static decoder;
- `D4 - D1`: attention beyond a similarly sized static decoder; and
- `D4 - D3`: attention versus recurrence at similar capacity.

Do not describe lower error from a larger decoder as improved representation
learning. External Raw LSTM and GARCH--LSTM results remain contextual complete-
system benchmarks because their input and optimization paths differ; only the
row-aligned D0--D4 comparisons isolate decoder effects.

## Tests and implementation order

Add focused tests under `tests/tasks/phase2_decoders/`:

1. row-map reconstruction, `K=8` shape, exact final-row alignment, and expected
   counts on a small multi-contract fixture;
2. rejection of cross-contract, cross-split, duplicate, reversed, skipped, or
   non-contiguous contexts;
3. split-safe price labels and rejection of global-array boundary shifts;
4. D0--D4 shape/dtype behavior, causal masking, deterministic inference,
   Softplus non-negativity, and exact parameter counts;
5. train-only standardizer provenance and identical D0--D4 target/identity
   hashes;
6. checkpoint snapshot/reload equivalence, prediction replay, non-finite
   failure, and no test inference before training completion;
7. bootstrap manifest expansion, overwrite/partial-run safeguards, and the
   expected 30 Stage-1 trajectories; and
8. report aggregation, contract-paired intervals, and failure on identity or
   source-checksum mismatch.

Implementation order:

1. data contracts and tests;
2. D0--D4 models and parameter-count tests;
3. single-run trainer and CPU fixture smoke tests;
4. bootstrap/replay/reporting tools;
5. one-epoch CPU smoke for every D0--D4/task combination;
6. freeze the generated Stage-1 matrix manifest;
7. execute the CUDA matrix; and
8. write the decoder judgement, then update status documents from artifacts.

## Completion gate

Part 1 is complete only when:

- temporal and price bundles pass reconstruction, split-safety, and replay;
- all D0--D4 Stage-1 runs use identical final target rows within each task;
- all 30 trajectories and 90 snapshots are present and replay-verified;
- multi-seed, per-contract, paired-uncertainty, parameter, timing, and memory
  comparisons are reported;
- volatility predictions are nonnegative by construction;
- conclusions distinguish static capacity, gating, recurrence, and attention;
  and
- the report labels current-test findings as characterization evidence.

Classification is a supported later extension, but it is not required to begin
or finish the price/volatility Stage-1 matrix.
