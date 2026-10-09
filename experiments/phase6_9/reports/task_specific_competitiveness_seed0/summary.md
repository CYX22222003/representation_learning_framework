# Phase 6.9 Task-Specific Competitiveness — Seed 0

**Status:** complete
**Principal checkpoint:** epoch 50, fixed before evaluation
**Scope:** two global-calendar walks; task-separated complete-system evidence

Phase 6.9 asks whether one reusable canonical `H0` representation remains
competitive when classification, future price, and future realised variance
are each compared with a task-oriented method. The three comparators are
heterogeneous and their metrics or ranks are not averaged.

## Classification: SGN-C

Every method below uses the original full h2/`tau=0.001` identities.

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

SGN-C trails H0-D0, Raw LSTM, and Raw MLP on principal macro-F1 in both
walks. Its hard variable assignment collapses all five variables into one
group at both primary checkpoints. Walk 2 macro-F1 falls from `0.444708` at
epoch 5 to `0.313716` at epoch 50; epoch 5 is retained but not selected post
hoc. This does not establish a classification-specific advantage over the
reusable representation.

## Future price: XM-C8

Every method uses all original h8 price rows and the sole `close[t+8h]`
endpoint target.

| Walk | Model | MAE | RMSE | Implied-movement Spearman |
|---:|---|---:|---:|---:|
| 1 | XM-C8 | 0.003100233 | 0.011040398 | 0.328457 |
| 1 | H0-D0 | 0.007985399 | 0.018905834 | 0.147363 |
| 1 | Raw LSTM | 0.004500579 | 0.012151731 | 0.263878 |
| 2 | XM-C8 | 0.004759289 | 0.020759157 | 0.259229 |
| 2 | H0-D0 | 0.010053535 | 0.030072603 | 0.099988 |
| 2 | Raw LSTM | 0.006541574 | 0.022129913 | 0.146491 |

XM-C8 improves MAE and RMSE over H0-D0 and Raw LSTM in both walks. It does not
consistently beat the non-learned current-price persistence reference:
persistence has lower MAE in both walks and lower RMSE in Walk 2. This is
target-matched complete-system evidence, not an isolated sLSTM or
representation gain.

## Future realised variance: GARCH--LSTM

All methods use the strict observed eight-hour future-realised-variance rows.

| Walk | Model | MAE | MSE | Spearman |
|---:|---|---:|---:|---:|
| 1 | GARCH--LSTM | 0.000960824 | 0.000654354 | 0.445114 |
| 1 | H0-D0 | 0.000626190 | 0.000654515 | 0.482914 |
| 1 | Raw LSTM | 0.000666902 | 0.000654618 | 0.465298 |
| 2 | GARCH--LSTM | 0.001470010 | 0.000096312 | 0.554133 |
| 2 | H0-D0 | 0.000502040 | 0.000097194 | 0.571345 |
| 2 | Raw LSTM | 0.000540722 | 0.000096315 | 0.555140 |

The stack provides only marginal MSE improvements while worsening MAE and
Spearman in both walks. It does not establish broad volatility superiority.

## Overall judgement

The reusable canonical representation remains competitive but does not win
every task. A task-specialized benefit is clear for XM-C8 against the learned
price models, while SGN-C is weaker on classification and the GARCH--LSTM
stack offers only metric-specific volatility gains. The result supports the
project's distinction between reusable representation evidence and
task-specific complete-system evidence.

No universal SOTA, multi-seed robustness, significance, profitable-alpha, or
architecture-causality claim follows. Optional Phase 6.8, deferred Phase 6.6,
and frozen Phase 7A remain separate.

## Source reports

- Classification: `experiments/phase6_9/sgn_classification/reports/seed0/matched_comparison.md`
- Price: `experiments/phase6_9/xlstm_mixer_endpoint/reports/seed0/summary.md`
- Volatility: `experiments/phase6_5/reports/b_c_seed0/principal_results.md`
- SGN execution record: `docs/phase_plan/2026-10-10-phase-6-9-sgn-execution.md`
