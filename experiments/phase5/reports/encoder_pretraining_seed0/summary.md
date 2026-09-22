# Phase 5 Canonical Encoder Pretraining

All six seed-0 walk-specific trajectories completed 50 epochs. Epoch 50 was predeclared for downstream feature extraction.

| Walk | Encoder | Epoch-50 loss | Embedding std | Collapse warning |
|---:|---|---:|---:|:---:|
| 1 | vae | 0.10893485 | 0.05645689 | false |
| 1 | contrastive | 2.33701080 | 1.18781137 | false |
| 1 | byol | 0.07470790 | 0.44650105 | false |
| 2 | vae | 0.07066280 | 0.09024052 | false |
| 2 | contrastive | 2.25663392 | 2.99989295 | false |
| 2 | byol | 0.05124977 | 0.70031059 | false |

These are unsupervised training and representation-health diagnostics, not downstream performance results.
