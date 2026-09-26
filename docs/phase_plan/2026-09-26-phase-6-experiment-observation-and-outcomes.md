# Phase 6 Experiment Observations and Outcomes

**Date:** 2026-09-26  
**Status:** Temporal-encoder seed-0 task complete; current-round volatility
neural execution complete; confirmatory follow-ups remain open  
**Authority:** Interprets the frozen contracts in
`2026-09-22-phase-6-temporal-encoder-variants-plan.md`,
`2026-09-22-phase-6-volatility-forecasting-plan.md`, and
`2026-09-25-phase-6-volatility-model-matrix-freeze.md` without changing their
models, rows, targets, or epoch-50 decision.

## 1. Scope completed

Phase 6 executed the precommitted temporal-encoder comparison on the two
accepted one-hour global-calendar walks. The completed evidence contains:

- eight walk-specific temporal SSL encoder trajectories: contrastive and BYOL
  LSTM/Transformer candidates for each walk;
- checkpoints at epochs 5, 15, and 50, with epoch 50 fixed in advance as the
  downstream feature source;
- six task/walk master feature stores covering the 11 declared `H0`,
  substitution, heterogeneous-addition, and duplicate-CNN configurations;
- six replayed `H0` task/walk references and 60 new downstream trajectories
  across two-hour classification, eight-hour future price, and eight-hour
  future realised variance;
- checkpoint, prediction, metric, source, scaler, row-identity, and feature
  replay for all 66 task/walk/configuration entries; and
- two fixed-sample linear-CKA diagnostics and the complete per-walk comparison
  report.

The generated machine report records `valid=true`, 66 entries, 360 paired
differences, eight encoder validations, six feature-store validations, and
two CKA validations. The detailed artifacts are:

- `experiments/phase6/encoder_variants/reports/complete_seed0/report.md`;
- `experiments/phase6/encoder_variants/reports/complete_seed0/epoch50_by_walk.csv`;
- `experiments/phase6/encoder_variants/reports/complete_seed0/paired_differences.csv`;
- `experiments/phase6/encoder_variants/reports/complete_seed0/encoder_resources.csv`;
- `experiments/phase6/encoder_variants/diagnostics/cka/walk1.json`; and
- `experiments/phase6/encoder_variants/diagnostics/cka/walk2.json`.

## 2. Comparison validity

The comparison preserves the Phase 5/6 fairness contract:

1. Each walk has separately trained encoders and downstream heads using only
   information available before its calendar cutoff.
2. Every model within a task and walk uses identical ordered training and
   evaluation identities and targets.
3. Coordinate scaling is fitted on task-training features only and then frozen.
4. The matrix uses one uninterrupted 50-epoch trajectory, fixed snapshots at
   epochs 5/15/50, and epoch 50 as the predeclared principal result.
5. No validation split, early stopping, restart selection, or evaluation-
   driven configuration choice was used.
6. Substitutions are compared with `H0` at the same 445-dimensional width.
   Additions are compared with both `H0` and a 573-dimensional exact duplicate-
   CNN control.

Raw-OHLCV MLP and Raw LSTM comparisons use the same task rows and metric
definitions. Classification and future-price comparators come from the
replay-validated Phase 5 matched matrix. Volatility comparators were retrained
on the stricter Phase 6 H=8 future-realised-variance rows.

All results remain single-seed characterisation evidence. Values identified
below as "best observed" describe the completed matrix; they do not select a
model or revise the frozen plan.

## 3. Representation similarity

Centered linear CKA compares each epoch-50 temporal branch with its same-family
CNN on the first 4,096 identities in saved walk-specific encoder-training
order:

| SSL family and temporal backbone | Walk 1 CKA | Walk 2 CKA |
|---|---:|---:|
| Contrastive LSTM | 0.2304 | 0.2151 |
| Contrastive Transformer | 0.2010 | 0.2044 |
| BYOL LSTM | 0.2961 | 0.2076 |
| BYOL Transformer | 0.2088 | 0.1709 |

The temporal representations are therefore not simple linear copies of their
CNN counterparts. This establishes representation difference only. The low to
moderate CKA values do not establish usefulness or complementarity; those
questions require the matched downstream comparisons below.

## 4. Downstream observations

### 4.1 Two-hour movement classification

| Walk | Model or descriptive entry | Macro-F1 | Balanced accuracy |
|---:|---|---:|---:|
| 1 | Canonical `H0` | **0.4535** | 0.4658 |
| 1 | Best observed temporal substitution | 0.4530 (`HC-ST`) | 0.4684 (`HC-ST`) |
| 1 | Raw-OHLCV MLP | 0.4448 | 0.4609 |
| 1 | Raw LSTM | 0.4252 | **0.4694** |
| 2 | Canonical `H0` | 0.4532 | 0.4892 |
| 2 | Best observed temporal substitution | **0.4587** (`HB-ST`) | **0.4954** (`HC-ST`) |
| 2 | Raw-OHLCV MLP | 0.4325 | 0.4571 |
| 2 | Raw LSTM | 0.4463 | 0.4846 |

The frozen representation framework remains strongest against the raw learned
baselines on macro-F1. Temporal substitution improves both principal metrics
on Walk 2, but not Walk 1 macro-F1. Walk 1 Raw LSTM balanced accuracy exceeds
the best temporal value by only `0.0010`.

The evidence supports task-specific competitiveness, not a universal temporal-
backbone win. LSTM/Transformer additions also vary by family and walk and do
not consistently beat both `H0` and their same-width duplicate controls.

### 4.2 Eight-hour future-price prediction

Price-level results are:

| Walk | Model or descriptive entry | Price MAE | Price RMSE |
|---:|---|---:|---:|
| 1 | Canonical `H0` | 0.007985 | 0.018906 |
| 1 | Lowest-MAE temporal entry, `HC-AL` | 0.006933 | 0.015184 |
| 1 | Raw-OHLCV MLP | 0.006865 | 0.015613 |
| 1 | Raw LSTM | 0.004501 | 0.012152 |
| 1 | Current-price persistence | **0.002978** | **0.011872** |
| 2 | Canonical `H0` | 0.010054 | 0.030073 |
| 2 | Lowest-MAE temporal entry, `HB-AL` | 0.007888 | 0.024590 |
| 2 | Raw-OHLCV MLP | 0.011125 | 0.023488 |
| 2 | Raw LSTM | 0.006542 | 0.022130 |
| 2 | Current-price persistence | **0.004426** | **0.020395** |

Every learned temporal substitution/addition improves price-level MAE over
`H0` in both walks except the duplicate-width controls, but the Raw LSTM remains
the strongest learned comparator overall. Temporal features beat the Raw MLP
on Walk 1 RMSE and Walk 2 MAE, yet no learned model beats the causal current-
price persistence reference.

Movement-ranking diagnostics are more demanding:

| Walk | Best temporal implied-movement Spearman | Raw MLP | Raw LSTM |
|---:|---:|---:|---:|
| 1 | 0.1368 (`HB-AL`) | 0.1468 | **0.2639** |
| 2 | 0.1148 (`HC-AL`) | 0.0587 | **0.1465** |

The Raw LSTM also has higher mean cross-sectional Rank IC than every learned
temporal configuration: `0.2574` versus a temporal maximum of `0.1377` on Walk
1, and `0.1191` versus `0.1110` on Walk 2. The fixed last-hour reversal
reference remains strongest at `0.3012` and `0.2400`.

The clearest evidence for heterogeneous temporal information is therefore
price-level reconstruction: additions frequently improve over `H0` and their
same-width duplicate controls. That evidence does not extend reliably to
financial movement ranking, and it does not establish incremental value over
raw sequential modelling or the reversal reference.

### 4.3 Eight-hour future realised variance

Different metrics favor different models. The temporal rows below are a
descriptive metric-wise envelope, not one selected configuration.

| Walk | Model or descriptive entry | MAE | RMSE | Pearson | Spearman |
|---:|---|---:|---:|---:|---:|
| 1 | Canonical `H0` | 0.0006262 | 0.0255835 | 0.0281 | 0.4829 |
| 1 | Best observed temporal value | 0.0006252 | 0.0255818 | 0.0328 | **0.5095** |
| 1 | Raw-OHLCV MLP | 0.0006274 | **0.0255454** | **0.1245** | 0.4868 |
| 1 | Raw LSTM | 0.0006669 | 0.0255855 | 0.0255 | 0.4653 |
| 1 | Historical persistence | 0.0010663 | 0.0355669 | 0.0339 | **0.5586** |
| 2 | Canonical `H0` | 0.0005020 | 0.0098587 | 0.0016 | 0.5713 |
| 2 | Best observed temporal value | 0.0005047 | 0.0098571 | 0.0110 | 0.5778 |
| 2 | Raw-OHLCV MLP | **0.0004964** | 0.0098551 | 0.0210 | **0.5784** |
| 2 | Raw LSTM | 0.0005407 | **0.0098140** | **0.1117** | 0.5551 |
| 2 | Historical persistence | 0.0005667 | 0.0100255 | 0.0052 | **0.6168** |

Walk 1 temporal features provide the strongest learned Spearman ordering but
the Raw MLP retains lower RMSE and much higher Pearson correlation. On Walk 2,
the Raw MLP leads MAE and narrowly leads Spearman, while the Raw LSTM leads
RMSE and Pearson. Historical persistence has the strongest Spearman ordering
in both walks but substantially worse level errors.

The absolute error differences among `H0`, temporal variants, and Raw MLP are
very small relative to the target tail. No temporal architecture dominates
across error, linear correlation, rank correlation, and both walks. The result
supports retaining multiple metrics and the historical-persistence reference;
it does not support choosing an encoder from one volatility number.

## 5. Architecture and resource observations

Transformer candidates cost materially more without consistent downstream
benefit:

| SSL family | LSTM trainable parameters | Transformer trainable parameters | Transformer/LSTM | Pretraining-time ratio, Walks 1/2 |
|---|---:|---:|---:|---:|
| Contrastive | 102,144 | 299,008 | 2.93x | 1.66x / 1.55x |
| BYOL | 135,168 | 332,032 | 2.46x | 1.74x / 1.67x |

Peak encoder-training CUDA memory was approximately 406--407 MB for LSTMs,
543 MB for the contrastive Transformer, and 582 MB for the BYOL Transformer.
Downstream additions increase the task head from approximately 65k to 82k
parameters because input width grows from 445 to 573.

Given the inconsistent downstream changes, the additional Transformer cost is
not justified as a default replacement by this seed-0 experiment. This is a
practical resource observation, not a parameter-matched architecture study.

## 6. Overall outcome

The Phase 6 evidence supports the following statements:

1. LSTM and Transformer branches learn representations that are linearly
   distinct from the corresponding CNN branches.
2. The frozen multi-branch framework is strongest against raw learned
   baselines for movement-classification macro-F1.
3. Temporal features substantially improve the framework's future-price level
   reconstruction, and heterogeneous additions often beat duplicate-width
   controls on that narrow capability.
4. A Raw LSTM still predicts future-price levels and ranks implied movement
   more effectively, while simple persistence/reversal references remain
   stronger than all learned models on their respective diagnostics.
5. Future-realised-variance results are metric- and walk-dependent. Temporal
   features sometimes improve rank ordering, but no architecture consistently
   beats Raw MLP, Raw LSTM, and historical persistence.
6. Transformers require roughly 2.5--3 times the trainable encoder parameters
   and 1.5--1.7 times the pretraining time of LSTMs without a repeatable
   downstream advantage.

The experiment therefore demonstrates task-specific representation value, not
universal temporal-encoder or framework superiority. It does not justify
selecting one temporal backbone for every downstream task.

## 7. Limitations and follow-ups

- Results use seed 0 only; small metric differences may be initialization
  noise. Multiple seeds or predeclared paired block intervals are required for
  strong architecture-ranking claims.
- Stride-one decision rows and overlapping horizons are temporally dependent.
- The duplicate-CNN width controls are perfectly collinear, making them useful
  but imperfect capacity controls.
- Linear CKA measures only linear representational similarity. A nonlinear
  kernel analysis may be added later as a separately motivated diagnostic,
  but is not needed to interpret the present downstream evidence.
- The causal reversal result was identified after evaluation and requires a
  fresh later holdout plus cost, liquidity, and source-orientation checks.
- Retrospective cohort selection, accepted offline cleaning, and condition-
  candle token orientation remain external-validity limitations.
- The adapted GARCH--LSTM comparator remains outside the completed current-
  round neural matrix and is now scoped separately by
  `2026-09-26-phase-6-5-lstm-capacity-and-garch-lstm-plan.md`.
- The volatility parent plan requested pooled reporting for every model. The
  current temporal report is complete per walk, while the initial H0/raw report
  contains the existing pooled comparison. A pooled artifact for all temporal
  configurations remains a reporting follow-up and must not be used for post-
  hoc model selection.

No additional configuration should be chosen from these evaluation results.
If confirmatory architecture ranking becomes necessary, the next defensible
step is a predeclared multi-seed rerun of a scientifically justified subset,
not a rerun of only the observed winners.

The approved immediate follow-ups are instead the bounded seed-0 LSTM-depth
capacity study and strict H=8 GARCH--LSTM benchmark in the Phase 6.5 plan, plus
the canonical branch-attribution matrix in
`2026-09-26-phase-7a-representation-ablation-plan.md`. Phase 7B alpha research
remains unspecified pending further literature review.
