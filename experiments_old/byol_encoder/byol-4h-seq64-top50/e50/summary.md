# BYOL encoder epoch 50

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- target decay: `0.99`
- final train BYOL loss: `0.0776185841`
- final view cosine: `0.9611907314`
- final embedding std: `0.9908381116`
- final embedding norm: `11.9647306540`
- collapse warning: `False`
- best observed train BYOL loss: `0.0039418837`
- elapsed seconds: `635.14`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
