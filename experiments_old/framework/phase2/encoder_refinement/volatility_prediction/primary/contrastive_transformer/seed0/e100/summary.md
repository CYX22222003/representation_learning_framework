# Framework volatility prediction epoch 100

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `data/features/phase2/five_branch_contrastive_transformer_4h_seq64_top50_seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0000101352`
- MAE: `0.0387329794`
- RMSE: `0.0970483050`
- MSE: `0.0094183739`
- Pearson correlation: `0.7215292454`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
