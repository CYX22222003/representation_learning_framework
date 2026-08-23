# Framework trend classification epoch 15

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50.npz`
- seed: `0`
- mode: `concat`
- train samples: `109741`
- test samples: `27400`
- final train loss: `0.4115946293`
- accuracy: `0.4730291963`
- macro-F1: `0.3874601574`
- weighted-F1: `0.4529548612`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `confusion_matrix.npz`, `metrics.json`.
