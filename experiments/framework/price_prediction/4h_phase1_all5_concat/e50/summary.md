# Framework price prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0005146248`
- MAE: `0.0650772005`
- RMSE: `0.0992591083`
- MSE: `0.0098523702`
- Pearson correlation: `0.9786238670`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
