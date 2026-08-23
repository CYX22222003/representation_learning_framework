# Aggregation

This package contains the representation fusion module used by the framework.

`RepresentationAggregator` accepts named feature branches such as
`statistical`, `transformed`, `vae`, and `contrastive`, then combines them into
a single embedding for downstream task heads. It supports concat fusion for the
MVP path and gated fusion for later comparison experiments.
