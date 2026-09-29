# Phase 6.5B/C Seed-0 Principal Results

**Generated:** 2026-09-29
**Principal snapshot:** epoch 50, fixed before evaluation
**Status:** all Phase 6.5B and 6.5C trajectories and 5/15/50 snapshots pass
their standalone replay validators. This is the compact principal-result
summary; the plan's contract-macro and subgroup reporting remains a follow-up.

## Execution evidence

- Phase 6.5B: two walk-specific GARCH--LSTM stacks, each with five
  chronological Raw-LSTM OOF folds. Replay-valid OOF populations are 25,174
  rows for Walk 1 and 44,874 rows for Walk 2.
- Phase 6.5C: two causal 36-feature stores and ten trajectories. The common
  train/test intersections contain 30,043/25,821 rows in Walk 1 and
  49,243/10,923 rows in Walk 2.
- Validation commands:
  - `.venv/bin/python3 scripts_v5/validate_phase6_5_garch_lstm.py`
  - `.venv/bin/python3 scripts_v5/validate_phase6_5_ta_mlp_data.py`
  - `.venv/bin/python3 scripts_v5/validate_phase6_5_ta_mlp.py`

## Phase 6.5B: H=8 realised variance

All values below use the unchanged Phase 6 evaluation identities and raw
realised-variance units.

| Walk | Model | MAE | MSE | Pearson | Spearman |
|---:|---|---:|---:|---:|---:|
| 1 | GARCH--LSTM stack | 0.000960824 | 0.000654354 | 0.025369 | 0.445114 |
| 1 | Raw LSTM | 0.000666902 | 0.000654618 | 0.025512 | 0.465298 |
| 1 | canonical H0 | 0.000626190 | 0.000654515 | 0.028056 | 0.482914 |
| 2 | GARCH--LSTM stack | 0.001470010 | 0.000096312 | 0.111526 | 0.554133 |
| 2 | Raw LSTM | 0.000540722 | 0.000096315 | 0.111689 | 0.555140 |
| 2 | canonical H0 | 0.000502040 | 0.000097194 | 0.001646 | 0.571345 |

The stack lowers MSE against Raw LSTM by only 0.0403% in Walk 1 and 0.0028%
in Walk 2. It lowers MSE against H0 by 0.0246% and 0.9075%, respectively, but
has worse MAE and Spearman in both walks. The result therefore does not support
a broad GARCH--LSTM win. It is a metric-specific MSE improvement accompanied
by worse absolute-error calibration and rank ordering.

No epoch-50 stack predictions were clipped at zero. The fitted ElasticNet
coefficients are stored with each walk's snapshot; they are descriptive and
do not establish standalone branch superiority.

## Phase 6.5C: h2/tau=0.001 movement classification

The primary comparison is the four P2 models on each walk's exact common
TA-eligible rows. P1U is shown separately as a TA-only training sensitivity.

| Walk | Model/protocol | Macro-F1 | Balanced accuracy | Macro ROC-AUC | Macro AP | NLL | Accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | H0/P2 | 0.447668 | 0.456570 | 0.723760 | 0.451770 | 0.855436 | 0.644088 |
| 1 | Raw MLP/P2 | 0.421119 | 0.419676 | 0.695915 | 0.435916 | 0.864199 | 0.675032 |
| 1 | Raw LSTM/P2 | 0.426742 | 0.463734 | 0.695897 | 0.432569 | 0.901507 | 0.558576 |
| 1 | TA-MLP/P2 | 0.480952 | 0.524718 | 0.735275 | 0.471719 | 0.812210 | 0.596336 |
| 1 | TA-MLP/P1U sensitivity | 0.483489 | 0.529587 | 0.741439 | 0.477527 | 0.833040 | 0.593780 |
| 2 | H0/P2 | 0.463227 | 0.495129 | 0.768657 | 0.482826 | 0.770262 | 0.613659 |
| 2 | Raw MLP/P2 | 0.442052 | 0.459017 | 0.750421 | 0.463652 | 0.792860 | 0.613293 |
| 2 | Raw LSTM/P2 | 0.455453 | 0.485961 | 0.766992 | 0.474370 | 0.777855 | 0.615033 |
| 2 | TA-MLP/P2 | 0.456409 | 0.467769 | 0.723506 | 0.468274 | 0.925558 | 0.629772 |
| 2 | TA-MLP/P1U sensitivity | 0.448849 | 0.476911 | 0.707615 | 0.451191 | 1.113664 | 0.586011 |

TA-MLP/P2 leads Walk 1 on the primary macro-F1 metric by 0.0333 over H0 and
also improves balanced accuracy, macro ROC-AUC, macro AP, and NLL. It does not
replicate that advantage in Walk 2: H0 leads macro-F1 by 0.0068 and is clearly
better on balanced accuracy, macro ROC-AUC, macro AP, NLL, and Brier score.
P1U is marginally better than TA-P2 on Walk 1 macro-F1 but worse on Walk 2;
it does not provide a stable sampling improvement.

The supported conclusion is task- and walk-specific: handcrafted TA features
are competitive and useful in Walk 1, but neither TA-MLP nor the representation
framework dominates both walks. These classification metrics are predictive
diagnostics, not evidence of profitable trading.

## Causal and replay amendments discovered before result interpretation

- OOF schema v2 discards a contract's fold rows when fewer than two causal
  prices exist at that fold's start. The same final OOF identities feed GARCH,
  Raw LSTM, and the meta-learner.
- An evaluation condition absent from a walk's training identities receives a
  deterministic fallback equal to the median guarded H=8 forecast across
  contract-local training-only GARCH states. Evaluation outcomes never update
  this fallback or any GARCH state.
- CUDA predictions are immutable metric inputs. Independent CPU replay uses
  bounded recurrent-backend tolerances only for checkpoint verification; it
  does not replace saved predictions or alter reported metrics.
