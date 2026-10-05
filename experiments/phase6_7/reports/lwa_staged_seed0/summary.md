# Phase 6.7 staged LWA-Frozen result

**Status:** LWA method matrix complete and valid

**Seed:** 0

**Principal checkpoint:** epoch 50

**Scope:** two walks, three tasks, native 384-dimensional representation

This is the retained method-local LWA result. The required SaURL matrix and
optional TimeDART extension are also complete. The final cross-method
interpretation is in
`experiments/phase6_7/reports/frozen_representation_seed0/summary.md`.

## Artifact completeness

- Both training-only FFT/CWT caches pass their source, scaler, dependency,
  shape, and content-hash contracts.
- Both walk-specific 50-epoch joint and 50-epoch mapper trajectories retain
  the frozen 5/15/50 checkpoints, histories, and replay records.
- Both 384-dimensional master stores preserve the established ordered task
  identities and pass feature validation.
- All six downstream trajectories contain valid epochs 5, 15, and 50, for 18
  evaluated snapshots.
- Scale-small Wavelet CPU/CUDA replay differences are retained as warnings
  under the predeclared relative-L2/cosine bounds; no structural, provenance,
  non-finite, artifact-integrity, or material-drift failure was accepted.

## Epoch-50 principal results

### Movement classification

| Walk | Method | Accuracy | Macro-F1 | Balanced accuracy |
|---:|---|---:|---:|---:|
| 1 | LWA-F | 0.5430 | 0.4031 | 0.4689 |
| 1 | H0 | 0.6480 | 0.4535 | 0.4658 |
| 2 | LWA-F | 0.6135 | 0.4497 | 0.4837 |
| 2 | H0 | 0.6127 | 0.4532 | 0.4892 |

LWA is materially weaker on Walk 1 accuracy and macro-F1. Walk 2 accuracy is
effectively tied, while macro-F1 and balanced accuracy remain below H0.

### Absolute future price

| Walk | Method | MAE | RMSE | Price Pearson | Price Spearman | Implied-movement Pearson | Implied-movement Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LWA-F | 0.012594 | 0.023817 | 0.9934 | 0.9686 | 0.1803 | 0.1079 |
| 1 | H0 | 0.007985 | 0.018906 | 0.9962 | 0.9511 | 0.2094 | 0.1474 |
| 2 | LWA-F | 0.015669 | 0.028236 | 0.9893 | 0.9870 | -0.0040 | 0.0754 |
| 2 | H0 | 0.010054 | 0.030073 | 0.9875 | 0.9823 | -0.0035 | 0.1000 |

LWA has worse MAE in both walks and worse Walk 1 RMSE. It improves Walk 2
RMSE and price-level correlations, but implied-movement Spearman is lower in
both walks.

### Future realised variance

| Walk | Method | MAE | RMSE | Pearson | Spearman |
|---:|---|---:|---:|---:|---:|
| 1 | LWA-F | 0.000654 | 0.025589 | 0.0155 | 0.4430 |
| 1 | H0 | 0.000626 | 0.025583 | 0.0281 | 0.4829 |
| 2 | LWA-F | 0.000516 | 0.009808 | 0.1583 | 0.5450 |
| 2 | H0 | 0.000502 | 0.009859 | 0.0016 | 0.5713 |

LWA is weaker on every Walk 1 metric. In Walk 2 it improves RMSE and Pearson,
but has worse MAE and Spearman.

## Bounded judgement

Under the fixed native-width frozen-probing protocol, this paper-guided
independent LWA implementation does not consistently outperform canonical
`H0`. Its improvements are metric- and walk-specific and do not support a
broad representation-superiority claim. This remains seed-0, two-walk
characterisation evidence, not a universal or state-of-the-art claim.

The complete 5/15/50 metrics, predictions, scalers, histories, identities,
and hashes remain in the corresponding `encoder_pretraining/lwa_frozen/`,
`features/lwa_frozen/`, and `downstream/<task>/lwa_frozen/` artifact
directories. Large checkpoints, master arrays, transform caches, and runtime
logs remain persistent on Lumid and are intentionally Git-ignored.
