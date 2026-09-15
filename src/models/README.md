# Models

This package contains neural encoder architectures and their objective helpers.

The canonical SSL encoders are VAE, contrastive CNN, and BYOL. These models are
pretrained on the training split only, then frozen and used to extract named
embedding branches for the downstream framework evaluation.

Phase 2 adds basic `contrastive_lstm` and `contrastive_transformer` candidate
implementations in `encoder_variants.py`, backed by reusable sequence modules
in `temporal_backbones.py`. They keep the canonical NT-Xent augmentations,
projector semantics, and 128-dimensional downstream interface. They are named
experimental substitutions for `contrastive`, not replacements for its CNN
implementation. Full-data training and downstream evaluation remain pending.
