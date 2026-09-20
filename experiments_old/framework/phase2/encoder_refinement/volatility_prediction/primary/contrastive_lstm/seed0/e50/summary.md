# Framework volatility prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/phase2/five_branch_contrastive_lstm_4h_seq64_top50_seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0000138711`
- MAE: `0.0384522192`
- RMSE: `0.0941583887`
- MSE: `0.0088658025`
- Pearson correlation: `0.7368196845`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
