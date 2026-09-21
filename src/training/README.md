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

`phase5_downstream.py` owns the completed framework-only seed-0 probe: one
train-only feature standardizer per walk, fixed probability-point regression,
logit-adjusted tri-class classification, 5/15/50 artifacts, detailed
breakdowns and non-trained references, and CPU prediction replay. Learned
baseline comparisons are a separate pending Phase 5 stage.

`phase5_regression_sensitivities.py` owns the completed exploratory
eight-hour raw-change and two-hour log-return probes, including train-only
target transforms, probability reconstruction, price-band breakdowns, and
5/15/50 CPU prediction replay.
