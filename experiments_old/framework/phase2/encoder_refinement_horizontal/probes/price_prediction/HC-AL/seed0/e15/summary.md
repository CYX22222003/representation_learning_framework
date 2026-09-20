# Framework price prediction epoch 15

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `experiments/framework/phase2/encoder_refinement_horizontal/features/supersets/contrastive/seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0003824496`
- MAE: `0.0517256446`
- RMSE: `0.0788533315`
- MSE: `0.0062178480`
- Pearson correlation: `0.9869747758`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
