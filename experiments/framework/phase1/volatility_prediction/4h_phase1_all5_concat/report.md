# Framework Volatility Prediction Report

All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.

## Data and representation

- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- branches: `{'statistical': 70, 'transformed': 55, 'vae': 64, 'contrastive': 128, 'byol': 128}`
- train/test samples: `109791` / `27450`
- label bundle: `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz`

## Results

| epoch | MAE | RMSE | MSE | Pearson correlation |
|---:|---:|---:|---:|---:|
| 15 | 0.040595 | 0.089076 | 0.007935 | 0.767419 |
| 50 | 0.036816 | 0.086553 | 0.007491 | 0.769867 |
| 100 | 0.035934 | 0.087684 | 0.007688 | 0.763630 |

Images are stored under the run-level and per-budget `images/` folders.
