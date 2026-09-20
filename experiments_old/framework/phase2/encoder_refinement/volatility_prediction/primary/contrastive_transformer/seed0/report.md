# Framework Volatility Prediction Report

All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.

## Data and representation

- feature store: `data/features/phase2/five_branch_contrastive_transformer_4h_seq64_top50_seed0.npz`
- branches: `{'statistical': 70, 'transformed': 55, 'vae': 64, 'contrastive_transformer': 128, 'byol': 128}`
- train/test samples: `109791` / `27450`
- label bundle: `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz`

## Results

| epoch | MAE | RMSE | MSE | Pearson correlation |
|---:|---:|---:|---:|---:|
| 15 | 0.042793 | 0.096685 | 0.009348 | 0.718471 |
| 50 | 0.038765 | 0.095401 | 0.009101 | 0.734808 |
| 100 | 0.038733 | 0.097048 | 0.009418 | 0.721529 |

Images are stored under the run-level and per-budget `images/` folders.
