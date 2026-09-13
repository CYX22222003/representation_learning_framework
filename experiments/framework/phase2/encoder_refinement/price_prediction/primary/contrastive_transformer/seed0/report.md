# Framework Price Prediction Report

All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.

## Data and representation

- feature store: `data/features/phase2/five_branch_contrastive_transformer_4h_seq64_top50_seed0.npz`
- branches: `{'statistical': 70, 'transformed': 55, 'vae': 64, 'contrastive_transformer': 128, 'byol': 128}`
- train/test samples: `109840` / `27499`
- label bundle: `None`

## Results

| epoch | MAE | RMSE | MSE | Pearson correlation |
|---:|---:|---:|---:|---:|
| 15 | 0.056800 | 0.089423 | 0.007997 | 0.983085 |
| 50 | 0.056746 | 0.085600 | 0.007327 | 0.985316 |
| 100 | 0.061155 | 0.089794 | 0.008063 | 0.984106 |

Images are stored under the run-level and per-budget `images/` folders.
