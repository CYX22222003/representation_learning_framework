# Framework volatility prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0000182535`
- MAE: `0.0368162394`
- RMSE: `0.0865533873`
- MSE: `0.0074914885`
- Pearson correlation: `0.7698674798`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
