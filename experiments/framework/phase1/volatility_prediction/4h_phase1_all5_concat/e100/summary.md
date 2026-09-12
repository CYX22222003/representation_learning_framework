# Framework volatility prediction epoch 100

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0000115001`
- MAE: `0.0359340347`
- RMSE: `0.0876836926`
- MSE: `0.0076884301`
- Pearson correlation: `0.7636303902`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
