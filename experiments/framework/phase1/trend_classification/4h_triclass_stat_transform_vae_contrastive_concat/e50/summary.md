# Framework trend classification epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50.npz`
- seed: `0`
- mode: `concat`
- train samples: `109741`
- test samples: `27400`
- final train loss: `0.3890349567`
- accuracy: `0.4952554703`
- macro-F1: `0.4164474747`
- weighted-F1: `0.4812287516`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `confusion_matrix.npz`, `metrics.json`.
