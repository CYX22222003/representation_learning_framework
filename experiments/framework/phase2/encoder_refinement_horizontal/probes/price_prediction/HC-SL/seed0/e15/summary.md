# Framework price prediction epoch 15

Frozen feature branches are probed with the framework aggregator and a simple MLP head.

- processed dataset: `data/processed/market_4h_seq64_top50.npz`
- feature store: `experiments/framework/phase2/encoder_refinement_horizontal/features/supersets/contrastive/seed0.npz`
- seed: `0`
- mode: `concat`
- train samples: `109791`
- test samples: `27450`
- final train loss: `0.0004558045`
- MAE: `0.0499909893`
- RMSE: `0.0735261738`
- MSE: `0.0054060980`
- Pearson correlation: `0.9896601439`

Artifacts: `checkpoint.pth`, `history.npz`, `predictions.npz`, `metrics.json`.
