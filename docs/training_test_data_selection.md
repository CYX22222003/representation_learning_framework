# Training and Test Data Selection

This document specifies exactly which data subset each model component uses for training, validation, and evaluation. The rules here exist to prevent data leakage and ensure all models are compared fairly on the same held-out test set.

---

## The Global Split (already implemented)

The 80/20 chronological split applied per contract in `src/data_processing/data_processing.py` is the **only** split boundary that determines what is "seen" vs. "unseen" during model development.

```
Per-contract time series  →  first 80% = train portion
                          →  last  20% = test portion

After merging across all top-50 contracts:
  data/processed/*.npz  →  key "train"  shape [N_train, seq_len, 5]
                        →  key "test"   shape [N_test,  seq_len, 5]
```

**The test split is locked.** No model parameters — supervised or unsupervised — may be influenced by test sequences until the final evaluation run.

---

## Validation Split (for development)

A validation set is carved from the training split. Its purpose is monitoring training progress and early stopping. It is **never** used to make architectural decisions or report final results.

```
train split  →  first 80% = actual training data   (fit model parameters)
             →  last  20% = validation data         (monitor loss, early stopping)
```

Since sequences within the merged array are ordered by contract then chronologically within each contract, the last 20% of the training array approximately corresponds to the temporally later windows — a reasonable proxy for a held-out period.

In code, before any training loop:
```python
n = len(train_sequences)
val_split = int(0.8 * n)
train_data = train_sequences[:val_split]
val_data   = train_sequences[val_split:]
```

---

## Data Usage Per Component

| Component | Trains on | Validates on | Evaluated on |
|---|---|---|---|
| VAE encoder | train data | val data (reconstruction loss) | — (frozen after pretraining) |
| Contrastive encoder | train data | val data (NT-Xent loss) | — (frozen after pretraining) |
| Additional neural encoders (TBD) | train data | val data | — (frozen after pretraining) |
| Statistical features | *(deterministic — no fitting)* | — | — |
| Transformation features | *(deterministic — no fitting)* | — | — |
| RepresentationAggregator | train feature bundles | val feature bundles | test feature bundles |
| Task heads (price, volatility, trend) | train feature bundles | val feature bundles | test feature bundles |
| LSTM baseline | train sequences | val sequences | test sequences |
| Raw-OHLCV MLP baseline | train sequences | val sequences | test sequences |
| Single-branch ablations | train feature bundles | val feature bundles | test feature bundles |
| Standalone GARCH (volatility) | train sequences (fit per window) | — | test sequences |

---

## Feature Extraction for the Test Set

After neural encoders are trained and frozen, their weights are fixed. Running the frozen encoder on test sequences is **not leakage** — it is equivalent to applying a fitted scaler. The encoder parameters contain no information derived from test sequences.

```
Frozen VAE encoder + Frozen contrastive encoder
        │
        ├── inference on train sequences → train neural embeddings
        └── inference on test sequences  → test neural embeddings

Statistical + transform features computed independently for both splits.

Combined into FeatureBundle:
  data/features/*_train.npz  (statistical, transformed, neural arrays for train)
  data/features/*_test.npz   (same structure for test)
```

The `NpzFeatureStore` already handles saving/loading these bundles. The companion `.index.npz` stores `train_size` and `test_size` so the split can be recovered if bundles are concatenated.

---

## Order of Operations

The sequence below must be followed to avoid leakage.

```
1. data/processed/*.npz already exists
        │
        ▼
2. Determine val split index from train array
   (no model fitting at this step)
        │
        ▼
3. Pretrain VAE on train_data; monitor val_data reconstruction loss
   Save checkpoint: checkpoints/vae_<timeframe>.pth
        │
        ▼
4. Pretrain contrastive encoder on train_data; monitor val_data NT-Xent loss
   Save checkpoint: checkpoints/contrastive_<timeframe>.pth
        │
        ▼
   (repeat step 3–4 for any additional neural encoders — TBD)
        │
        ▼
5. Extract features for ALL sequences using frozen encoders:
     - statistical + transform: computed directly (no encoder needed)
     - neural branches: run frozen encoder inference
   Save: data/features/*_train.npz and data/features/*_test.npz
        │
        ▼
6. Train RepresentationAggregator + task head on train feature bundles
   Monitor val feature bundles; save best checkpoint per task
        │
        ▼
7. Train all baseline models on train sequences (or train feature bundles
   for single-branch ablations); same val split for monitoring
        │
        ▼
8. ── FINAL EVALUATION (one time only) ──
   Load all checkpoints; run inference on test feature bundles / test sequences
   Record metrics for every model × every task
```

---

## Aggregator Mode and Task Head Input Dimension

`RepresentationAggregator` supports two modes (set at construction time):

- **`mode="concat"` (default)**: branches are concatenated; `output_dim = sum of branch dims`. No learnable parameters in the aggregator itself — the task head does all the supervised learning.
- **`mode="gated"`**: each branch is projected to `out_dim`, then a gating network produces softmax weights. `output_dim = out_dim`. Adds learnable parameters; useful as a comparison once concat baseline results are available.

In both modes, `agg.output_dim` returns the correct input dimension for the task head:

```python
agg = RepresentationAggregator(branch_dims, mode="concat")
task_head = nn.Linear(agg.output_dim, n_outputs)  # works for both modes
```

Both modes are trained on the same data splits and evaluated identically, making them directly comparable in the ablation study.

---

## Rules Summary

1. **Split once, at the start.** The 80/20 per-contract split is already done. Do not re-split.
2. **Unsupervised ≠ exempt from the split.** VAE and contrastive encoders are trained on `train_data` only, never on test sequences.
3. **Val split is for monitoring only.** No architectural choice or hyperparameter may be made by looking at test metrics.
4. **All baselines use the identical train/val/test partitions** as the framework — same `.npz` files, same split indices.
5. **Test set is evaluated once**, after all development is complete. Re-running on test to chase metrics invalidates the comparison.
6. **Frozen encoder inference on test sequences is valid.** Encoder weights are fixed; no test-set gradient flows back.
