# Phase 6.7 staged SaURL-TS-Frozen result

**Status:** staged SaURL-only matrix complete and standalone replay-valid

**Seed:** 0

**Principal checkpoint:** epoch 50

**Scope:** two walks, three tasks, native 128-dimensional representation

This is an interim method-local result. It does not complete Phase 6.7: LWA
remains required, and these metrics may not change its later implementation.

## Artifact completeness

- Both walk-specific epoch-50 feature stores replay exactly on CUDA and retain
  the exact established task identities.
- All six SaURL downstream trajectories contain replay-valid epochs 5, 15,
  and 50, for 18 evaluated snapshots.
- All six immutable `H0` task/walk references pass standalone replay.
- SaURL feature extraction is target-independent; every task scaler is fitted
  only on that task's supervised training embeddings.

## Epoch-50 principal results

### Movement classification

| Walk | Method | Accuracy | Macro-F1 | Balanced accuracy |
|---:|---|---:|---:|---:|
| 1 | SaURL-F | 0.5835 | 0.4293 | 0.4630 |
| 1 | H0 | 0.6480 | 0.4535 | 0.4658 |
| 2 | SaURL-F | 0.6100 | 0.4503 | 0.4883 |
| 2 | H0 | 0.6127 | 0.4532 | 0.4892 |

SaURL is weaker on all three principal classification metrics in both walks.
The gap is small in Walk 2 and materially larger for Walk 1 accuracy and
macro-F1.

### Absolute future price

| Walk | Method | MAE | RMSE | Price Pearson | Price Spearman | Implied-movement Pearson | Implied-movement Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | SaURL-F | 0.009226 | 0.017709 | 0.9968 | 0.9386 | 0.2119 | 0.1341 |
| 1 | H0 | 0.007985 | 0.018906 | 0.9962 | 0.9511 | 0.2094 | 0.1474 |
| 2 | SaURL-F | 0.013524 | 0.031302 | 0.9864 | 0.9829 | 0.0078 | 0.0873 |
| 2 | H0 | 0.010054 | 0.030073 | 0.9875 | 0.9823 | -0.0035 | 0.1000 |

SaURL improves Walk 1 RMSE and marginally improves some Pearson diagnostics,
but has worse MAE in both walks, worse Walk 2 RMSE, and lower implied-movement
Spearman in both walks. Both learned methods remain worse than persistence on
price-level MAE.

### Future realised variance

| Walk | Method | MAE | RMSE | Pearson | Spearman |
|---:|---|---:|---:|---:|---:|
| 1 | SaURL-F | 0.000647 | 0.025587 | 0.0213 | 0.4546 |
| 1 | H0 | 0.000626 | 0.025583 | 0.0281 | 0.4829 |
| 2 | SaURL-F | 0.000536 | 0.009821 | 0.0959 | 0.5581 |
| 2 | H0 | 0.000502 | 0.009859 | 0.0016 | 0.5713 |

SaURL is weaker on MAE and Spearman in both walks. It has a small Walk 2 RMSE
improvement and a larger Walk 2 Pearson improvement, but these do not support
consistent volatility superiority.

## Bounded judgement

Under the fixed native-width frozen-probing protocol, this SaURL-TS
paper-guided reimplementation does not consistently outperform canonical
`H0`. Its isolated improvements are metric- and walk-specific. The result is
seed-0, two-walk characterisation evidence only, not a universal or
state-of-the-art claim. Near-zero learned SaDA masks and large pretraining
embedding norms remain relevant implementation diagnostics when interpreting
the downstream result.

The complete 5/15/50 metrics, predictions, scalers, histories, identities,
and hashes remain in the corresponding `features/saurl_frozen/` and
`downstream/<task>/saurl_frozen/` artifact directories.
