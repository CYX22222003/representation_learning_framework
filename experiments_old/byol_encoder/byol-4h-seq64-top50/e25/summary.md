# BYOL encoder epoch 25

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- target decay: `0.99`
- final train BYOL loss: `0.0796773641`
- final view cosine: `0.9601613389`
- final embedding std: `0.7765034167`
- final embedding norm: `10.5953616029`
- collapse warning: `False`
- best observed train BYOL loss: `0.0039418837`
- elapsed seconds: `315.87`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
