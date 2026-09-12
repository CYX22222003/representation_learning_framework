# Framework volatility prediction epoch 15

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0000443274`
- MAE: `0.0405952185`
- RMSE: `0.0890758410`
- MSE: `0.0079345061`
- Pearson correlation: `0.7674192786`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
