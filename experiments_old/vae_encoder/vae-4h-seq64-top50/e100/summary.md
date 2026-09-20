# VAE encoder epoch 100

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- beta: `1.0`
- final total loss: `0.0801516505`
- final reconstruction MSE: `0.0625848766`
- final KL divergence: `0.0175667738`
- best observed total loss: `0.0800388432`
- elapsed seconds: `125.33`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
