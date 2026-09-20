# Framework Price Prediction Report

All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.

## Data and representation

- feature store: `data/features/phase2/five_branch_contrastive_lstm_4h_seq64_top50_seed0.npz`
- branches: `{'statistical': 70, 'transformed': 55, 'vae': 64, 'contrastive_lstm': 128, 'byol': 128}`
- train/test samples: `109840` / `27499`
- label bundle: `None`

## Results

| epoch | MAE | RMSE | MSE | Pearson correlation |
|---:|---:|---:|---:|---:|
| 15 | 0.051663 | 0.076882 | 0.005911 | 0.988570 |
| 50 | 0.056498 | 0.079488 | 0.006318 | 0.989040 |
| 100 | 0.060262 | 0.083837 | 0.007029 | 0.987802 |

Images are stored under the run-level and per-budget `images/` folders.
