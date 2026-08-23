# Framework price prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0005326392`
- MAE: `0.0694802403`
- RMSE: `0.1045774966`
- MSE: `0.0109364530`
- Pearson correlation: `0.9763363600`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
