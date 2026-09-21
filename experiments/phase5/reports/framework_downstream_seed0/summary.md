# Phase 5 Five-Branch Framework Downstream Probe (Seed 0)

Epoch 50 was predeclared. Walks were trained and evaluated independently; pooled metrics were computed only after both out-of-future prediction sets existed.

## Epoch-50 results by walk

| Walk | Regression MAE | Regression RMSE | Pearson | Spearman | Classification macro-F1 | Balanced accuracy | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00255587 | 0.00929351 | 0.112727 | 0.001235 | 0.453469 | 0.465787 | 0.647956 |
| 2 | 0.00305107 | 0.01360845 | 0.030296 | -0.005724 | 0.453202 | 0.489162 | 0.612659 |

## Pooled out-of-future predictions

- Regression framework: MAE 0.00271136, RMSE 0.01083505, Pearson 0.080745, Spearman -0.000518.
- Regression exact-zero reference: MAE 0.00224972, RMSE 0.01077716.
- Classification framework: macro-F1 0.454116, balanced accuracy 0.472953, accuracy 0.636873.
- Always-STABLE reference: macro-F1 0.283121, balanced accuracy 0.333333, accuracy 0.738169.
- Walk-local training-prior reference: macro-F1 0.283121, balanced accuracy 0.333333, accuracy 0.738169.

These are framework-only seed-0 probing results, not final comparative claims. Learned baselines and additional seeds remain deferred.
