# Models

This package contains neural encoder architectures and their objective helpers.

The current SSL encoders are VAE, contrastive CNN, and BYOL. These models are
pretrained on the training split only, then frozen and used to extract named
embedding branches for the downstream framework evaluation.
