# Phase-1 Product Readiness Plan

Date: 2026-09-01

## Purpose and terminology

This plan prepares the **Phase-1 Product**: the first complete framework
product using all currently available representation branches:

```text
statistical + transformed + VAE + contrastive + BYOL
```

The earlier four-branch work remains the **MVP**. It established that the
feature-store, frozen-representation probing, price task, and trend task work
end to end while BYOL downstream extraction and the strict volatility
benchmarks were still unavailable. Phase 1 supersedes that MVP configuration
for new experiments; it does not invalidate or overwrite its recorded results.

Phase 1 is not yet the final research evaluation. Branch ablations,
multi-seed confirmation, gated fusion, decoder-controlled comparisons, and
cross-timeframe transfer remain later Phase-C work.

## Current evidence

Ready artifacts:

| Item | Evidence |
|---|---|
| Processed 4h dataset | `data/processed/market_4h_seq64_top50.npz`; 109,841 train and 27,500 test sequences |
| VAE encoder | `checkpoints/vae_4h_seq64_top50.pth` |
| Contrastive encoder | `checkpoints/contrastive_4h_seq64_top50.pth` |
| BYOL encoder | `checkpoints/byol_4h_seq64_top50.pth`; epoch 100, seq-len 64, input dim 5, embedding dim 128 |
| Strict volatility labels | `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz`; 109,791 train and 27,450 test aligned rows |
| Volatility external benchmarks | Raw LSTM and adapted GARCH--LSTM stacking runs on that shared label bundle |

Current blockers:

1. `scripts/prepare_framework_features.py` extracts only VAE and contrastive
   embeddings; the saved 4h store contains no `byol` branch.
2. `scripts/train_framework.py` accepts only price prediction and trend
   classification; it cannot consume the shared volatility label contract.
3. `scripts/validate_feature_store.py` does not make BYOL a required default
   branch, so a missing BYOL array could pass the old validation path.

## Fixed Phase-1 decisions

- Scope: 4h, sequence length 64, top-50 contracts.
- Fusion: `RepresentationAggregator(mode="concat")`.
- Full embedding size: `70 + 55 + 64 + 128 + 128 = 445`.
- Downstream tasks: price prediction, trend classification, and volatility
  prediction.
- Training protocol: fixed seed `0`; fixed epoch budgets `15,50,100`; one
  uninterrupted trajectory to epoch 100 and saved snapshots.
- Run names must be new, explicit Phase-1 names. Do not use `--overwrite` on
  earlier MVP run directories.
- The test partition is evaluated only after all fixed-budget snapshots are
  trained. No validation split, early stopping, or test-driven selection.

## Task 1: add frozen BYOL extraction to the feature builder

**Files:**

- Modify `scripts/prepare_framework_features.py`
- Extend or create focused tests for the feature builder

**Implementation:**

1. Import `BYOLEncoder` from `models.byol`.
2. Add `--byol-checkpoint`, defaulting to
   `checkpoints/byol_4h_seq64_top50.pth`.
3. Load and validate the checkpoint's sequence length and input dimension using
   the existing checkpoint-contract guard.
4. Construct `BYOLEncoder` from checkpoint metadata (`input_dim`,
   `hidden_dim`, `projection_dim`, and `predictor_hidden_dim`) and load the
   model state.
5. In inference mode, call `model.encode(batch)` to save the online-backbone
   representation, not the projector, predictor, or target network output.
6. Store it under the independent branch name `byol` for train and test; add
   provenance, output shape, and embedding-source details to the manifest.
7. Preserve the existing named VAE and contrastive branches and deterministic
   extraction behaviour.

**Acceptance:** A regenerated feature store has exactly these five named
arrays, all `float32`, finite, and split-aligned:

```text
statistical  [137341,  70]
transformed  [137341,  55]
vae          [137341,  64]
contrastive  [137341, 128]
byol         [137341, 128]
```

## Task 2: make feature-store validation enforce Phase-1 branches

**Files:**

- Modify `scripts/validate_feature_store.py`
- Add/update validator tests if present

**Implementation:**

1. Add `byol: 128` to the default expected dimensions.
2. Retain `--allow-extra-branches` only as an explicit compatibility escape
   hatch; Phase-1 validation must use the exact five-branch set.
3. Verify manifest shapes, companion split index, processed split sizes,
   finite values, and `float32` dtypes.
4. Add an actionable error if the new BYOL branch is absent or dimensionally
   incompatible.

**Acceptance:** The validator rejects the existing four-branch store under the
Phase-1 default and accepts the regenerated five-branch store.

## Task 3: add framework volatility prediction

**Files:**

- Modify `scripts/train_framework.py`
- Use `src/tasks/volatility_labels.py` and
  `src/tasks/volatility_prediction.py`
- Add focused tests for label selection and task dispatch

**Implementation:**

1. Add `volatility_prediction` to CLI choices and `FrameworkConfig` task
   validation.
2. Require `--labels-npz` for volatility as well as trend classification.
3. Load the shared bundle through its validation loader rather than rebuilding
   targets with the legacy merged-array helper.
4. Select framework rows strictly using `train_row_indices` and
   `test_row_indices`; pair them respectively with `train_labels` and
   `test_labels`.
5. Validate that indices are in the matching split, labels are finite and
   non-negative, and row counts agree for every branch.
6. Add the `VolatilityRegressor` head, regression target kind, MSE loss, and
   MAE/RMSE/MSE/Pearson-correlation reporting.
7. Persist the label path and label manifest in the dataset manifest so the
   run is auditable against Raw LSTM and GARCH--LSTM stacking.

**Acceptance:** A small CPU smoke run writes normal framework snapshot
artifacts for volatility, and its prediction target arrays have exactly 109,791
train / 27,450 test rows with the shared-bundle identities.

## Task 4: regression and contract tests

Run the smallest relevant test modules after each implementation task, then
the complete targeted suite before GPU work. At minimum, test:

- BYOL checkpoint loading and extraction shape/dtype/finiteness;
- feature-store branch names, index integrity, and manifest contents;
- price and trend runner behaviour unchanged with a five-branch store;
- volatility runner uses bundle indices rather than recomputed merged targets;
- failures for missing/mismatched label bundles, branches, or checkpoint
  contracts;
- snapshot paths, configurations, and metrics for all three tasks.

Do not start the long Phase-1 run until those checks pass.

## Task 5: construct and validate the Phase-1 feature store

Run after Tasks 1--4 are complete:

```bash
.venv/bin/python3 scripts/prepare_framework_features.py \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --out-path data/features/features_4h_seq64_top50_phase1.npz \
  --vae-checkpoint checkpoints/vae_4h_seq64_top50.pth \
  --contrastive-checkpoint checkpoints/contrastive_4h_seq64_top50.pth \
  --byol-checkpoint checkpoints/byol_4h_seq64_top50.pth \
  --device cuda \
  --batch-size 1024

.venv/bin/python3 scripts/validate_feature_store.py \
  --features-npz data/features/features_4h_seq64_top50_phase1.npz \
  --processed-npz data/processed/market_4h_seq64_top50.npz
```

The new filename deliberately preserves the four-branch MVP store and makes
Phase-1 provenance unambiguous.

Before training, record the validation output and inspect the manifest's three
checkpoint paths and branch shapes. The statistical branch's known extreme
finite GARCH-derived values should be retained as a diagnostic; standardisers
must fit on train rows only and be saved with each downstream run.

## Task 6: execute the predeclared Phase-1 matrix

The full matrix is specified in
`docs/superpowers/specs/2026-09-01-phase-1-product-experiment-spec.md`.

Phase-1 completion requires all three task runs and their saved artifacts. A
run may be reported as a characterization result, but no epoch budget may be
called the selected winner after reading locked-test metrics.

The completed Phase-1 price run used the then-current merged-array horizon-1
helper (109,840 train / 27,499 test rows). It is preserved as legacy
characterisation evidence. New Phase-2 price studies use the contract-safe
109,791/27,450-row label bundle, which removes every contract's terminal row;
cross-generation claims require a contract-safe rerun. See
`docs/price_prediction_label_contract.md`.

## Work deliberately deferred

- Single-branch and leave-one-branch-out ablations;
- concat versus gated fusion;
- multi-seed estimates and paired-bootstrap intervals;
- strict trend alignment for TA-MLP and strict price alignment for the stacked
  LSTM;
- transferability and alpha-factor research.

The Raw-OHLCV MLP volatility baseline must be rerun with the shared label
bundle before any strict Phase-1 framework-vs-MLP volatility claim. It is not
a blocker for training the framework itself, but it is a blocker for that
specific comparison claim.
