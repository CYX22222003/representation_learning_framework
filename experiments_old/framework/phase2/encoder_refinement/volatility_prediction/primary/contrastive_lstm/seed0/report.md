# Framework Volatility Prediction Report

All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.

## Data and representation

- feature store: `data/features/phase2/five_branch_contrastive_lstm_4h_seq64_top50_seed0.npz`
- branches: `{'statistical': 70, 'transformed': 55, 'vae': 64, 'contrastive_lstm': 128, 'byol': 128}`
- train/test samples: `109791` / `27450`
- label bundle: `data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz`

## Results

| epoch | MAE | RMSE | MSE | Pearson correlation |
|---:|---:|---:|---:|---:|
| 15 | 0.041920 | 0.094035 | 0.008843 | 0.736905 |
| 50 | 0.038452 | 0.094158 | 0.008866 | 0.736820 |
| 100 | 0.038014 | 0.094670 | 0.008962 | 0.731162 |

Images are stored under the run-level and per-budget `images/` folders.
