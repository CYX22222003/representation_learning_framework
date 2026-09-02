# Framework price prediction epoch 100

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0003879796`
- MAE: `0.0678433403`
- RMSE: `0.1008787751`
- MSE: `0.0101765273`
- Pearson correlation: `0.9784035683`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
