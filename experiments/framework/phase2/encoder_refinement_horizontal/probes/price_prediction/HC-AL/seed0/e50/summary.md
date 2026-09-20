# Framework price prediction epoch 50

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `experiments/framework/phase2/encoder_refinement_horizontal/features/supersets/contrastive/seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0001992127`
- MAE: `0.0522157028`
- RMSE: `0.0740292221`
- MSE: `0.0054803258`
- Pearson correlation: `0.9906011820`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
