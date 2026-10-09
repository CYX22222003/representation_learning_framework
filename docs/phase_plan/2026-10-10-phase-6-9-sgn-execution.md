# Phase 6.9 SGN-C Execution and Outcome

**Date:** 2026-10-10
**Status:** classification leg complete for the frozen seed-0 scope
**Method:** independently authored supervised SGN-C
**Primary checkpoint:** epoch 50, fixed before evaluation

## 1. Completion record

Both walk-specific SGN-C trajectories completed 50 epochs using every original
h2/`tau=0.001` classification row. Snapshots at epochs 5, 15, and 50 retain
complete model, optimizer, assignment-temperature, sampler, and RNG state.
All six checkpoints, predictions, identities, and metrics pass same-backend
CUDA replay.

| Walk | Train/evaluation rows | Training time | Epoch-50 loss | Peak CUDA allocation |
|---:|---:|---:|---:|---:|
| 1 | 37,864 / 30,340 | 43.99 min | 0.878870 | 752.28 MiB |
| 2 | 57,521 / 13,887 | 65.44 min | 0.832644 | 752.28 MiB |

The admitted training implementation fingerprint remains
`7b52bff866a9adceb0b343246d0e665eb6e53e356eee080aceaf93ffb37ed57e`.

## 2. SGN-C snapshot results

| Walk | Epoch | Accuracy | Macro-F1 | Balanced accuracy | Hard groups O/H/L/C/V |
|---:|---:|---:|---:|---:|---|
| 1 | 5 | 0.487871 | 0.353539 | 0.456371 | `[1,1,1,1,0]` |
| 1 | 15 | 0.571457 | 0.360281 | 0.465832 | `[1,1,1,1,1]` |
| 1 | 50 | 0.585959 | 0.398941 | 0.442285 | `[1,1,1,1,1]` |
| 2 | 5 | 0.590480 | 0.444708 | 0.484238 | `[1,1,1,1,1]` |
| 2 | 15 | 0.536113 | 0.425283 | 0.486780 | `[1,1,1,1,1]` |
| 2 | 50 | 0.410960 | 0.313716 | 0.356374 | `[1,1,1,1,1]` |

Walk 2 is strongest at epoch 5, but epoch 50 remains the primary result. The
evaluation does not retroactively select epoch 5 or authorize tuning.

## 3. Principal matched comparison

| Walk | Model | Accuracy | Macro-F1 | Balanced accuracy |
|---:|---|---:|---:|---:|
| 1 | SGN-C | 0.585959 | 0.398941 | 0.442285 |
| 1 | H0-D0 | 0.647956 | 0.453469 | 0.465787 |
| 1 | Raw LSTM | 0.542057 | 0.425204 | 0.469376 |
| 1 | Raw MLP | 0.628378 | 0.444806 | 0.460938 |
| 2 | SGN-C | 0.410960 | 0.313716 | 0.356374 |
| 2 | H0-D0 | 0.612659 | 0.453202 | 0.489162 |
| 2 | Raw LSTM | 0.596313 | 0.446326 | 0.484626 |
| 2 | Raw MLP | 0.613739 | 0.432459 | 0.457089 |

Pooled epoch-50 macro-F1 is `0.373931` for SGN-C, versus `0.454116` for
H0-D0, `0.430663` for Raw LSTM, and `0.441711` for Raw MLP.

## 4. Diagnostics and judgement

SGN-C does not beat H0-D0 on principal macro-F1 or balanced accuracy in either
walk. It also trails Raw LSTM on both primary metrics in both walks and trails
Raw MLP on macro-F1 in both walks. The evidence therefore does not establish a
classification-specific advantage over the reusable canonical representation.

The train-only initializer separated OHLC from volume in both walks, but the
learned hard assignment collapsed all variables into one group by epoch 15 in
Walk 1 and by epoch 5 in Walk 2. This removes the intended inter-group pathway
at the primary checkpoints and is an important negative method diagnostic.
It is retained rather than tuned away after evaluation.

Walk 2 also shows substantial fixed-budget degradation between epochs 5 and
50. This is evidence of overfitting or distribution mismatch under the frozen
recipe, not permission to select an earlier checkpoint.

## 5. Replay note

Same-backend CUDA replay is exact enough to pass all structural, provenance,
prediction, identity, and metric checks. The first CPU replay exposed benign
cross-device convolution drift at Walk 1 epoch 5: maximum absolute logit drift
`1.47e-5`, relative L2 `1.03e-5`, cosine `1.0`. The launcher now uses a
post-training same-backend validator without changing the admitted training
implementation or checkpoints.

## 6. Claim boundary

This is a two-walk, seed-0 complete-system comparison. SGN-C receives direct
classification supervision and has a native head, so the result is not a
frozen-representation or architecture-isolating comparison. It supports no
universal, multi-seed, significance, or trading claim.

Principal artifacts are under:

```text
experiments/phase6_9/sgn_classification/
```

The matched report is
`experiments/phase6_9/sgn_classification/reports/seed0/matched_comparison.md`.
