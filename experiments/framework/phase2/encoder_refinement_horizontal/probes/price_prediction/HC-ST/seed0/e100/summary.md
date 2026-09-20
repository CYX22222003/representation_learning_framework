# Framework price prediction epoch 100

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `experiments/framework/phase2/encoder_refinement_horizontal/features/supersets/contrastive/seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0001919695`
- MAE: `0.0510189198`
- RMSE: `0.0785160214`
- MSE: `0.0061647659`
- Pearson correlation: `0.9874038100`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
