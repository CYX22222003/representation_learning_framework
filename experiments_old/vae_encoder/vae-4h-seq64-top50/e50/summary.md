# VAE encoder epoch 50

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- beta: `1.0`
- final total loss: `0.0827743671`
- final reconstruction MSE: `0.0647486225`
- final KL divergence: `0.0180257445`
- best observed total loss: `0.0809418668`
- elapsed seconds: `64.44`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
