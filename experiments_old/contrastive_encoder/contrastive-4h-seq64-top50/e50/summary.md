# Contrastive encoder epoch 50

This is an unsupervised pretraining run. The held-out test split is not used for training or model selection.

- dataset: `data/processed/market_4h_seq64_top50.npz`
- seed: `0`
- device: `cuda`
- train sequences: `109841`
- batch size: `256`
- temperature: `0.2`
- final train NT-Xent loss: `2.4600282077`
- best observed train NT-Xent loss: `2.4599065508`
- elapsed seconds: `727.67`

Artifacts: `checkpoint.pth`, `history.npz`, `metrics.json`.
