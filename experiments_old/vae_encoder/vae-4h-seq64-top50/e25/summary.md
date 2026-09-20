# VAE encoder epoch 25

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- beta: `1.0`
- final total loss: `0.0841178878`
- final reconstruction MSE: `0.0659496934`
- final KL divergence: `0.0181681944`
- best observed total loss: `0.0809418668`
- elapsed seconds: `32.58`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
