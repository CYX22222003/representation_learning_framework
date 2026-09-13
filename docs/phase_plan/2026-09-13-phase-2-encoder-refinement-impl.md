# Phase 2 Encoder Refinement Implementation Plan

Date: 2026-09-08

## Scope and isolation

This branch owns Part 2 of the Phase 2 experiment plan. The canonical
`contrastive` and `byol` CNN branches remain unchanged. The primary work adds
`contrastive_lstm` and `contrastive_transformer` as 128-dimensional frozen
backbone substitutions while keeping the NT-Xent objective, Phase-1
augmentations, projector semantics, processed data, and downstream shallow
probe fixed.

The BYOL LSTM and Transformer variants are intentionally deferred until the
predeclared BYOL-only and all-minus-BYOL ablation gate is evaluated.

## Implementation stages

### Stage E1 — basic candidate setup (implemented, not trained)

- Add reusable LSTM and compact Transformer sequence backbones.
- Add named contrastive candidate construction with a common 128-dimensional
  downstream interface.
- Reuse `models.contrastive.make_views` and `nt_xent_loss` without changing
  their definitions.
- Add finite-loss/gradient checks and representation-collapse diagnostics.
- Add a fixed-budget training entry point that writes new Phase 2 artifacts
  and records architecture, dataset checksum, parameter count, and timing.
- Add frozen embedding extraction with checkpoint and data provenance.
- Cover model contracts, one-epoch CPU training, checkpointing, and extraction
  with focused tests.

The command defaults implement the general plan's starting designs: one-layer
LSTM with hidden size 128, and a two-layer Transformer with model dimension
128, four heads, feed-forward dimension 256, dropout 0.1, sinusoidal position
encoding, post-norm encoder layers, and final-token readout. These are
executable defaults, not a frozen final matrix. The experiment specification
must approve or revise them before task-test evaluation.

### Stage E2 — experiment specification (later)

- Freeze architecture values, seeds, epoch budgets, optimizer settings,
  collapse thresholds, and resource-reporting rules using training-side
  diagnostics only.
- Record the CNN reference parameter count and define the acceptable capacity
  comparison language; do not tune candidates from downstream test metrics.
- Freeze run names and the complete candidate-by-seed matrix.

### Stage E3 — pretraining and frozen features (later)

- Run CPU smoke tests for both candidates, then uninterrupted CUDA trajectories.
- Generate all fixed-budget checkpoints and training diagnostics.
- Extract frozen train/test branch artifacts under `data/features/phase2/` and
  replay their checksums, shapes, finite values, and row counts.

### Stage E4 — controlled downstream evaluation (later)

- Run each variant alone with the fixed shallow task head.
- Substitute exactly one candidate for `contrastive` in the five-branch bundle;
  do not add CNN and temporal variants together in the primary comparison.
- Use identical eligible target rows for price, shared-label volatility, and
  probability-movement classification.
- Compare each candidate with the immutable CNN reference across the complete
  fixed-budget, multi-seed matrix and report resource costs.

## Basic commands

```bash
.venv/bin/python3 scripts/train_phase2_contrastive_encoder.py \
  --variant contrastive_lstm \
  --run-name contrastive_lstm-4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --device cuda

.venv/bin/python3 scripts/train_phase2_contrastive_encoder.py \
  --variant contrastive_transformer \
  --run-name contrastive_transformer-4h-seq64-top50-seed0 \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --device cuda

.venv/bin/python3 scripts/extract_phase2_encoder_features.py \
  --checkpoint checkpoints/phase2/contrastive_lstm-4h-seq64-top50-seed0.pth \
  --out-path data/features/phase2/contrastive_lstm_4h_seq64_top50_seed0.npz \
  --device cuda
```

No full-data training or downstream task-test evaluation is part of Stage E1.
