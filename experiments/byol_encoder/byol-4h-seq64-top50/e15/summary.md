# BYOL encoder epoch 15

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- target decay: `0.99`
- final train BYOL loss: `0.0754657454`
- final view cosine: `0.9622671515`
- final embedding std: `0.6184719251`
- final embedding norm: `9.3672230383`
- collapse warning: `False`
- best observed train BYOL loss: `0.0039418837`
- elapsed seconds: `191.52`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
