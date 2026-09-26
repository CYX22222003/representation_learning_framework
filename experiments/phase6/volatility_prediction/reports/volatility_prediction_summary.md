# Phase 6 Volatility Prediction Summary

**Status:** Partial. The corrected H=8 volatility task, data, features, and six
ready neural trajectories are complete and replay-validated. The active
current-round matrix has 26 trajectories: six complete and 20 temporal/control
trajectories pending. Adapted GARCH--LSTM is deferred to a later round and is
not a current-round completion condition.

## Scientific contract

- Target: `RV(t,8h) = sum_{j=1..8} (p[t+j] - p[t+j-1])^2` on raw prediction-
  market probabilities over the strictly future interval `(t,t+8h]`.
- Context: 64 hourly candles. Every target-path candle is observed,
  consecutive, and in the same uninterrupted segment; accepted isolated
  one-hour fills may occur only in historical context and are recorded.
- Walk 1: 32,470 training and 27,806 evaluation rows.
- Walk 2: 53,112 training and 12,115 evaluation rows.
- The two 445-dimensional canonical feature stores are exact identity
  selections from frozen Phase 5 features and do not use volatility targets.
- Training: seed 0, 50 uninterrupted epochs, snapshots at 5/15/50, fixed
  `10,000 * RV` optimization units, Smooth L1 loss, and nonnegative Softplus
  output. Epoch 50 was predeclared as the principal snapshot.

## Completed current-round evidence

H0 canonical framework, Raw-OHLCV MLP, and Raw LSTM are complete for both
walks. All 18 saved checkpoints replay their evaluation predictions on CPU.

| Walk | Model | MAE | RMSE | Pearson | Spearman |
|---:|---|---:|---:|---:|---:|
| 1 | Canonical framework H0 | 0.00062619 | 0.02558349 | 0.0281 | 0.4829 |
| 1 | Raw-OHLCV MLP | 0.00062744 | 0.02554544 | 0.1245 | 0.4868 |
| 1 | Raw LSTM | 0.00066690 | 0.02558550 | 0.0255 | 0.4653 |
| 2 | Canonical framework H0 | 0.00050204 | 0.00985872 | 0.0016 | 0.5713 |
| 2 | Raw-OHLCV MLP | 0.00049637 | 0.00985513 | 0.0210 | 0.5784 |
| 2 | Raw LSTM | 0.00054072 | 0.00981402 | 0.1117 | 0.5551 |

The exact-zero MAE is 0.00066158 in Walk 1 and 0.00051293 in Walk 2.
Historical persistence has worse error than H0 and Raw MLP in both walks, but
the strongest rank correlation: 0.5586 in Walk 1 and 0.6168 in Walk 2.

In the descriptive pooled view, Raw MLP has the best learned-model MAE, RMSE,
and Pearson correlation; H0 has the best learned-model Spearman correlation.
This is not a model-selection result because the 20 temporal/control
trajectories have not run and the walks remain the primary comparison units.

## Interpretation

The corrected target is feasible and the initial learned models slightly
improve error over the exact-zero reference, but their error advantage is
small. Historical persistence ranks volatility better while producing much
worse absolute errors, which indicates a difficult heavy-tailed calibration
problem. The current evidence supports target validity and pipeline readiness;
it does not yet support framework superiority or a final volatility model.

## Remaining current-round work

1. Verify the temporal encoder trainer and launcher end to end.
2. Train eight walk-specific temporal encoders.
3. Extract the variant/control feature stores and run the 20 pending
   volatility heads.
4. Replay all 26 trajectories and produce the complete per-walk, strata, and
   tail-concentration report.

GARCH--LSTM may be reconsidered only in a later round with a separate audited
adapter and chronological out-of-fold stacking contract.
