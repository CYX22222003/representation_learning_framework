# Framework Trend Classification Report

All fixed epoch budgets are retained as a characterization sweep; no checkpoint is selected from locked-test performance.

## Data and representation

- feature store: `data/features/features_4h_seq64_top50_phase1.npz`
- branches: `{'statistical': 70, 'transformed': 55, 'vae': 64, 'contrastive': 128, 'byol': 128}`
- train/test samples: `109741` / `27400`
- label bundle: `data/task_labels/trend_classification/triclass_4h_seq64_top50.npz`

## Results

| epoch | accuracy | macro-F1 | weighted-F1 |
|---:|---:|---:|---:|
| 15 | 0.455000 | 0.375672 | 0.438402 |
| 50 | 0.476095 | 0.394168 | 0.459460 |
| 100 | 0.476460 | 0.393482 | 0.459768 |

Images are stored under the run-level and per-budget `images/` folders.
