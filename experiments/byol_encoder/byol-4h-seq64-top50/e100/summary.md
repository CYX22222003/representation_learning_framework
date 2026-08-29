# BYOL encoder epoch 100

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- target decay: `0.99`
- final train BYOL loss: `0.0526101089`
- final view cosine: `0.9736949717`
- final embedding std: `1.3356986524`
- final embedding norm: `13.6477851512`
- collapse warning: `False`
- best observed train BYOL loss: `0.0039418837`
- elapsed seconds: `1283.12`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
