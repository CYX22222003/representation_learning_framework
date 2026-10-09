# XM-MV8 Walk 1 — seed 0

Completed 50 epochs on 2026-10-09 using local WSL, vanilla sLSTM on CUDA,
and the dedicated `.venv-xlstm-mixer` environment. Checkpoint, prediction,
identity, and metric replay pass at all three frozen snapshots.

This model receives full five-channel OHLCV supervision at horizons t+1..t+8;
the headline extracts `close[t+8]`. It is a contextual complete-system baseline,
not a target-matched representation control. Epoch 50 was fixed before evaluation.

| Epoch | Future-close MAE | RMSE | Implied-movement Spearman | Full-path MAE |
|---:|---:|---:|---:|---:|
| 5 | 0.003332264 | 0.011668802 | 0.272780 | 0.015411984 |
| 15 | 0.003194807 | 0.011078153 | 0.316926 | 0.014685248 |
| 50 | 0.003054837 | 0.010852719 | 0.352519 | 0.014379537 |

Training/evaluation rows: 32,470/27,786; evaluation contracts: 23.
Fixed recipe: seed 0, batch 512 (64 batches, final batch 214), Adam 1e-4,
no validation, early stopping, scheduler, AMP, or test-driven checkpoint selection.
Parameters: 94,992; summed epoch training time: 79.064 seconds (excludes
preflight, serialization, evaluation, and replay); peak training CUDA allocation:
298,847,744 bytes; epoch-50 inference time: 0.381 seconds.

At epoch 50, persistence MAE/RMSE are 0.002993605/0.012199731: XM-MV8
has 2.05% worse MAE but 11.04% lower RMSE. Implied-movement Pearson is
0.459719, sign agreement 0.463183, and mean timestamp-level Rank IC 0.327667
over 1,816 eligible timestamps. Last-hour reversal Spearman is 0.310013.
Contract-macro MAE/RMSE are 0.003119514/0.005383454.

Unclipped full-path diagnostics are retained: invalid OHLC ordering in 20.26%
of paths, negative inverse-transformed raw volume in 13.90%, and probability
outside [0,1] in 0%. No post-hoc constraint repair or retraining was applied.

Detailed artifacts: [metrics](e50/metrics.json), [replay](replay_validation.json),
[all snapshots](sweep_metrics.json), and [completion](training_complete.json).
Matched H0-D0 and Raw-LSTM intersection reruns remain required before any
learned-comparator competitiveness claim.
