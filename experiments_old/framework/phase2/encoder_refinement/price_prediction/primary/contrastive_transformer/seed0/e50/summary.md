# Framework price prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/phase2/five_branch_contrastive_transformer_4h_seq64_top50_seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0003467697`
- MAE: `0.0567464381`
- RMSE: `0.0855996683`
- MSE: `0.0073273033`
- Pearson correlation: `0.9853163958`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
