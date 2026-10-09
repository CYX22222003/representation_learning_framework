# Phase 6.9 XM-C8 endpoint execution

**Date:** 2026-10-09. **Status:** Price leg complete; Phase 6.9 classification
remains open. Authority: [endpoint amendment](2026-10-09-phase-6-9-xlstm-endpoint-amendment.md).

Both fresh seed-0 trajectories completed exactly 50 epochs on the local RTX
4060 Laptop GPU with vanilla sLSTM in `.venv-xlstm-mixer/`. Each uses all
original price rows, historical `[64,5]` contexts, one direct `close[t+8h]`
target, endpoint-only MSE, and the frozen batch-512/Adam-1e-4 recipe.
Epoch 50 is principal; no evaluation occurred before trajectory completion.
All six 5/15/50 snapshots pass training-end and independent same-backend
checkpoint/prediction/identity/target/history/metric replay.

| Walk | Method | Evaluation rows | MAE | RMSE |
|---:|---|---:|---:|---:|
| 1 | XM-C8 | 29,834 | 0.003100233 | 0.011040398 |
| 1 | H0-D0 | 29,834 | 0.007985399 | 0.018905834 |
| 1 | Raw LSTM | 29,834 | 0.004500579 | 0.012151731 |
| 1 | Persistence | 29,834 | 0.002978262 | 0.011871641 |
| 2 | XM-C8 | 13,506 | 0.004759289 | 0.020759157 |
| 2 | H0-D0 | 13,506 | 0.010053535 | 0.030072603 |
| 2 | Raw LSTM | 13,506 | 0.006541574 | 0.022129913 |
| 2 | Persistence | 13,506 | 0.004426176 | 0.020395383 |

XM-C8 improves MAE/RMSE over both learned controls in both walks. Persistence
still wins MAE in both walks and RMSE in Walk 2. Implied-movement Spearman is
0.328457/0.259229; timestamp cross-sectional Rank IC is 0.306897/0.230508.
No epoch-50 predictions lie outside `[0,1]`; outputs were not clipped.
Training-loop times are 97.60/128.95 seconds; peak allocated GPU memory is
284.12/284.22 MiB. These exclude data auditing, evaluation, checkpoint I/O,
and independent replay overhead.

## Reporting-only recovery

The original execution log is
`experiments/phase6_9/xlstm_mixer_endpoint/logs/pipeline_20261009T123409_35225.log`.
The launcher completed both trajectories and both replay-valid completion
markers, but its final comparison writer exited 1 with `KeyError: spearman`:
H0 stores movement statistics under `implied_movement.overall`; the other
methods store them directly under `implied_movement`.

The separate `scripts_v8/report_phase6_9_xlstm_endpoint.py` reads both native
schemas, reaudits original controls and frozen code/data/runtime identities,
and independently replays all six snapshots before writing the report.
Two reporting regression tests pass. No fingerprinted training code,
checkpoint, prediction, or metric file was rewritten and no guard was relaxed.
The complete focused suite passes 45 tests with one opt-in historical GPU
fixture skipped; real-data selected-backend admission and replay ran on CUDA.
The shell launcher now replays/reports completed runs without retraining; its
fresh-run recovery recognizes only this reporting error and still requires
complete artifacts and full independent validation. Subsequent launcher
validation exits zero in
`experiments/phase6_9/xlstm_mixer_endpoint/logs/pipeline_20261009T124344_36463.log`.

Admitted training fingerprint:
`c73aaec2009988802ec30fd0099f1ce0b97e30879fe519cc8897735c235e29ce`.
Artifacts and principal comparison:
`experiments/phase6_9/xlstm_mixer_endpoint/reports/seed0/summary.{md,json}`.
Both run roots contain configurations, manifests, full epoch histories,
resumable states, three snapshots, predictions, metrics, replay evidence,
and completion markers. Historical XM-MV8 artifacts remain unchanged.

## Claim boundary

This is a target/row/loss-matched complete-system comparison, not isolated
representation or architecture causality. The inverse-RevIN output differs
from sigmoid controls and gradient clipping is retained only for XM-C8.
The endpoint amendment followed review of XM-MV8 results, so this is an
explicit post-result protocol correction. One seed on two existing walks
does not establish significance, universal superiority, or profitable alpha.
Fresh holdouts or separately approved replications are needed for strong
confirmatory generalization claims. No decoder study or other task was launched.
