# Framework Price Prediction Report

All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.

## Data and representation

- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- branches: `{'statistical': 70, 'transformed': 55, 'vae': 64, 'contrastive': 128, 'byol': 128}`
- train/test samples: `109840` / `27499`
- label bundle: `None`

## Results

| epoch | MAE | RMSE | MSE | Pearson correlation |
|---:|---:|---:|---:|---:|
| 15 | 0.051227 | 0.090831 | 0.008250 | 0.979379 |
| 50 | 0.065077 | 0.099259 | 0.009852 | 0.978624 |
| 100 | 0.067843 | 0.100879 | 0.010177 | 0.978404 |

Images are stored under the run-level and per-budget `images/` folders.
