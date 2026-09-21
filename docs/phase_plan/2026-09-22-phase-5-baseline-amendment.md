# Phase 5 Matched Raw-Baseline Amendment

**Date:** 2026-09-22  
**Status:** Implemented and unit-tested; matrix not frozen on disk and no training executed  
**Parent authority:** `2026-09-21-phase-5-experiment-plan.md`

## 1. Scope

The learned Phase 5 baseline stage contains exactly two raw-sequence models,
three tasks, and two independently fitted calendar walks:

| Baseline | Role |
|---|---|
| `raw_ohlcv_mlp` | Internal end-to-end baseline over the flattened 64-by-5 context |
| `raw_ohlcv_lstm` | Three-layer temporal benchmark over the ordered 64-by-5 context |

The tasks are the original two-hour movement regression, the original
two-hour `DOWN/STABLE/UP` classification, and the eight-hour absolute future-
price regression. This is a 12-run seed-0 matrix. TA-MLP, TCN, additional
seeds, eight-hour direct-change regression, and log-return regression are not
part of this stage.

## 2. Shared data and leakage contract

Both baselines consume the saved `train_sequences` and `test_sequences`
directly from the validated Phase 5 task bundle. They do not rebuild windows,
targets, splits, activity eligibility, or gap segments.

- Two-hour tasks use
  `experiments/phase5/data_preparation/walk{1,2}/market_1h_seq64_h2.npz`.
- Absolute price uses
  `experiments/phase5/downstream_addons/shared/h8/data/walk{1,2}/market_1h_seq64_h8.npz`.
- OHLC remains in raw probability units.
- Volume uses the bundle's walk-training-interval-only z-score.
- Every learned model is fitted independently for each walk.
- Evaluation rows never affect preprocessing, fitting, checkpoint choice, or
  architecture.
- Saved train/evaluation identity hashes are copied into every run manifest,
  and all prediction files retain the original identities.

The h2 framework, Raw-OHLCV MLP, and raw LSTM therefore use identical rows and
targets. The h8 framework and both raw baselines likewise use the identical h8
population. No baseline-specific row filtering is allowed.

## 3. Frozen architectures

### Raw-OHLCV MLP

The internal baseline uses the existing `RawOHLCVMLP` design:

```text
flatten(64 x 5)
  -> 512 -> 512 -> 256 -> 256 -> 128
```

Hidden layers use ReLU and dropout `0.1`; the 128-dimensional output feeds the
same task-head family as the framework.

### Raw-OHLCV LSTM

The temporal benchmark is a leakage-safe multivariate adaptation of the
declared stacked LSTM:

```text
OHLCV[64,5]
  -> LSTM(5,50) -> dropout(0.2)
  -> LSTM(50,30) -> dropout(0.1)
  -> LSTM(30,20) -> final state -> dropout(0.05)
```

The final 20-dimensional state feeds the same task-head family. This runner is
new; the legacy close-only LSTM constructs labels before its split and is not
a Phase 5 comparator.

Both models use a `128 -> 64` GELU task head after their encoder output. The
classification head emits three logits. The absolute-price head ends in a
sigmoid and has no explicit current-price/persistence skip connection.

## 4. Frozen optimization contract

- seed: `0`;
- optimizer: Adam;
- learning rate: `1e-4`;
- weight decay: `0`;
- batch size: `512`;
- one uninterrupted 50-epoch trajectory;
- snapshots: epochs `5`, `15`, and `50`;
- principal result: epoch `50`, fixed before evaluation;
- no validation split, early stopping, or evaluation-driven selection.

Movement regression minimizes MSE in fixed probability-point units
`100 * delta` and inverts predictions before scientific metrics.
Classification uses natural training rows with train-prior logit-adjusted
cross-entropy (`lambda=1.0`), matching the framework. Absolute price minimizes
MSE directly in probability units with sigmoid-bounded output.

## 5. Evaluation and replay

Movement regression reports the framework's full MAE/RMSE/MSE,
Pearson/Spearman, sign, lifecycle, imputation, threshold, non-zero, and
contract-macro breakdowns, together with exact-zero reference metrics.

Classification reports macro-F1 and balanced accuracy as primary metrics and
the complete probability/classification metric payload on the untouched
evaluation distribution.

Absolute-price evaluation reports level metrics and breakdowns, persistence-
relative diagnostics, implied-movement metrics, and timestamp-level
cross-sectional Rank IC. The pooled report compares both baselines with the
already executed five-branch framework only after walk-local predictions have
been generated.

Every checkpoint retains configuration and source hashes. Validation reloads
all 5/15/50 checkpoints on CPU, replays predictions, verifies row identities,
and checks cumulative histories and completion markers.

## 6. Implementation and later execution

Reusable code:

```text
src/training/phase5_baselines.py
```

Entry points:

```bash
# Freeze the 12-run manifest and perform architecture smoke tests only.
.venv/bin/python3 scripts_v3/bootstrap_phase5_baselines.py --device cuda

# On the later execution day, train only missing runs.
.venv/bin/python3 scripts_v3/bootstrap_phase5_baselines.py --device cuda --execute
.venv/bin/python3 scripts_v3/validate_phase5_baselines.py
.venv/bin/python3 scripts_v3/report_phase5_baselines.py
```

The default bootstrap command does not train. Future artifacts belong under:

```text
experiments/phase5/baselines/
  manifests/baseline_matrix_seed0.json
  tasks/{task}/{baseline}/walk{1,2}/seed0/
  reports/baseline_matrix_seed0/
```

Implementation readiness is not experimental completion. Until the 12 runs
and report exist and pass replay validation, Phase 5 baseline results remain
pending and no cross-model performance claim is supported.
