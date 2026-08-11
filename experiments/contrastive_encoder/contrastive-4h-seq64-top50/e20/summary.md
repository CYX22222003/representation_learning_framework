# Contrastive encoder epoch 20

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- temperature: `0.2`
- final train NT-Xent loss: `2.5495741551`
- best observed train NT-Xent loss: `2.5495741551`
- elapsed seconds: `287.62`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
