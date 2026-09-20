# Framework price prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/phase2/five_branch_contrastive_lstm_4h_seq64_top50_seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0003039372`
- MAE: `0.0564982295`
- RMSE: `0.0794875324`
- MSE: `0.0063182684`
- Pearson correlation: `0.9890398383`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
