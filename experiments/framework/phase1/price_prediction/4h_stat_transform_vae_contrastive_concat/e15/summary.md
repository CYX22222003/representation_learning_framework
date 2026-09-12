# Framework price prediction epoch 15

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0010047419`
- MAE: `0.0574953221`
- RMSE: `0.0954704508`
- MSE: `0.0091146072`
- Pearson correlation: `0.9780145288`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
