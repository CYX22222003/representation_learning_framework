# Framework price prediction epoch 15

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/phase2/five_branch_contrastive_transformer_4h_seq64_top50_seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109840`
- test samples: `27499`
- final train loss: `0.0006225735`
- MAE: `0.0568001345`
- RMSE: `0.0894234404`
- MSE: `0.0079965517`
- Pearson correlation: `0.9830852151`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
