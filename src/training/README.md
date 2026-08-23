# Training

This package contains reusable training-loop functions for unsupervised encoder
pretraining.

The current loops cover VAE, contrastive, and BYOL encoders. Runnable experiment
entry points live in `scripts/`, while this package keeps the epoch-level
training logic shared and testable.
