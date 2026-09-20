# Framework price prediction epoch 100

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0002958881`
- MAE: `0.0500894077`
- RMSE: `0.0820917860`
- MSE: `0.0067390618`
- Pearson correlation: `0.9844262600`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
