# Training

This package contains reusable training-loop functions for unsupervised encoder
pretraining.

The current loops cover VAE, contrastive, and BYOL encoders. Runnable experiment
entry points live in `scripts/`, while this package keeps the epoch-level
training logic shared and testable.

`train_encoder_variants.py` provides the Phase 2 temporal-contrastive epoch
loop with non-finite gradient checks and representation-health diagnostics.
The fixed-budget entry point is `scripts/train_phase2_contrastive_encoder.py`.

# Phase 5

`phase5_encoder.py` owns reusable canonical VAE, contrastive-CNN, and BYOL-CNN
pretraining for the two global-calendar walks. It loads only
`encoder_train_sequences`, never downstream targets or evaluation values, and
stores independently initialized walk-specific checkpoints. Phase 5 executable
orchestration remains under `scripts_v3/`.
