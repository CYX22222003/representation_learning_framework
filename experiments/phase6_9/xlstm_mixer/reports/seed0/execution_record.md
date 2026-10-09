# XM-MV8 local training execution — 2026-10-09

Both independent seed-0 walks completed the frozen 50-epoch trajectory. All
six epochs-5/15/50 checkpoints, evaluation predictions, identities, and metrics
passed same-vanilla-backend CUDA replay. The complete launcher exited 0.
Epoch 50 was predeclared, not chosen using the evaluation results.

## Launch and runtime

Command: `bash scripts_v8/run_phase6_9_xlstm_mixer_experiment.sh --execute`.
Pipeline started 2026-10-09 11:28:18 UTC (19:28:18 Singapore); final log
write/completion was 11:34:47 UTC (19:34:47 Singapore), about 389 seconds later.
The durable log is `../../logs/pipeline_20261009T112818_31061.log`.

Local WSL used `.venv-xlstm-mixer/bin/python3`, Python 3.10.12, PyTorch
2.12.1+cu126, pinned xlstm 1.0.3 with the disclosed lazy CUDA-import patch,
and vanilla sLSTM on the RTX 4060 Laptop GPU (8 GiB). This is not execution
on the proposed remote Blackwell device. No nvcc/custom CUDA backend was used.
The admitted physical batch was 512, with limits 6 GiB and 600 seconds per
admission probe. Existing data, admission, and matrix were replayed unchanged.
The admission file's `training_launched: false` describes the admission-only
probe; it is deliberately immutable and is not the current trajectory status.

The recipe remained Adam 1e-4, no weight decay/scheduler, clipping norm 1,
float32 without AMP, L1 loss on all 8-by-5 targets, dropout 0.1, and exactly
one learned initial token. Both walks had independent weights and train-only
volume preprocessing. There was no validation, early stopping, or test selection.
Contexts are [64,5]; targets are fully observed [8,5]; headline evaluation
extracts `close[t+8]` and retains the original float64 endpoint labels.

## Predeclared epoch-50 results

XM-MV8 receives additional five-channel supervision at all eight future
horizons. This is a contextual complete-system baseline, not a target-matched
representation/architecture control. All prices/errors below use raw
probability units. Full-path MAE also includes train-normalized volume units.

| Diagnostic | Walk 1 | Walk 2 |
|---|---:|---:|
| Train / evaluation rows | 32,470 / 27,786 | 53,112 / 12,115 |
| Evaluation contracts | 23 | 25 |
| Future-close MAE | 0.003054837 | 0.004780288 |
| Future-close RMSE | 0.010852719 | 0.021858833 |
| Future-close MSE | 0.000117782 | 0.000477809 |
| Future-close Pearson / Spearman | 0.998633 / 0.983187 | 0.993122 / 0.992173 |
| Implied-movement Pearson / Spearman | 0.459719 / 0.352519 | 0.016708 / 0.257461 |
| Implied-movement sign agreement | 0.463183 | 0.499629 |
| Mean timestamp cross-sectional Rank IC | 0.327667 | 0.217973 |
| Eligible Rank-IC timestamps | 1,816 | 1,512 |
| Contract-macro MAE / RMSE | 0.003119514 / 0.005383454 | 0.005701409 / 0.013992779 |
| Full-path MAE | 0.014379537 | 0.016737754 |
| Persistence MAE | 0.002993605 | 0.004498082 |
| Persistence RMSE | 0.012199731 | 0.021350637 |
| MAE skill versus persistence | -2.05% | -6.27% |
| RMSE skill versus persistence | +11.04% | -2.38% |
| Last-hour reversal movement Spearman | 0.310013 | 0.258007 |
| Parameters | 94,992 | 94,992 |
| Sum of epoch training times (s) | 79.064 | 120.017 |
| Epoch-50 inference time (s) | 0.381 | 0.212 |
| Peak training CUDA allocation (bytes) | 298,847,744 | 298,917,376 |

The complete fixed sweep is in [summary.md](summary.md) and
[summary.json](summary.json). Both walks improve their endpoint errors and
global movement Spearman from epochs 5 to 50, but no intermediate checkpoint
was selected. Persistence MAE remains lower in both walks; XM-MV8 improves
RMSE only in Walk 1. Global movement ranking exceeds last-hour reversal in
Walk 1 and is effectively tied in Walk 2. High level correlation is not
independent evidence of incremental forecasting information.

Unconstrained output diagnostics are retained: invalid OHLC ordering affects
20.26%/54.17% of Walk 1/2 paths; inverse-transformed negative volume affects
13.90%/6.41%; neither walk predicts OHLC outside [0,1]. Fractions count paths
with at least one violating horizon/channel. Predictions are not clipped,
repaired, or used to retune the model.

## Replay and provenance

Data source/identity replay passed before training. Both complete run roots
contain config/model/environment/dependency/provenance records, the resumable
model/optimizer/RNG/sampler state, all snapshot histories, full-path predictions,
endpoint metrics, and completion/replay markers. Run-specific summaries are:

- `../../downstream/absolute_price_h8/walk1/xm_mv8/seed0/summary.md`
- `../../downstream/absolute_price_h8/walk2/xm_mv8/seed0/summary.md`

Implementation SHA-256:
`8b77c09e85cb5adeaacc3972b4aedb4910059e4745858e8ca63ce54a1e7b234a`.
Training-matrix SHA-256:
`b8407e49724f314783b6f588f2cbd4a58cfe215447a8caaad51f7c1b2a5b71fd`.

| Principal artifact | Walk 1 SHA-256 | Walk 2 SHA-256 |
|---|---|---|
| Epoch-50 checkpoint | `31bf2ce9510901a50022ba3fc7dc67631d9e956f1178d63534ad9371fd082d2f` | `2f7ded3d6a2220987814c9999a2b7216ae13e92811fdc867ebdcbc8d6cde5be3` |
| Epoch-50 predictions | `0556b4edd1bd58706f8fd53627ef2ec709380cc7daedd2eea4ce2227f84ac40f` | `774b4ef2c969a31213c30d65f1bd03eab92cb1c0e0036c15d654ec708e681068` |

## Remaining comparison scope

The observed-path eligibility reduces the broader historical h8 populations.
Matched H0-D0 and Raw-LSTM intersection **training reruns** remain required;
merely filtering their old predictions does not provide the frozen comparison.
No such controls were trained by this launch. Classification's inductive
Monotone-VI adaptation and the final Phase 6.9 comparative report remain open;
volatility reuses the completed Phase 6.5B stack. Phase 6.9 is not closed, and
these XM-only diagnostics support no universal-superiority or alpha claim.
