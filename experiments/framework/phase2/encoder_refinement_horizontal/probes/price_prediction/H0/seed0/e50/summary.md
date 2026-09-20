# Framework price prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0004158533`
- MAE: `0.0525269136`
- RMSE: `0.0836952627`
- MSE: `0.0070048971`
- Pearson correlation: `0.9845401049`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
