# Framework price prediction epoch 100

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0004076131`
- MAE: `0.0720483065`
- RMSE: `0.1058754772`
- MSE: `0.0112096164`
- Pearson correlation: `0.9760525823`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
