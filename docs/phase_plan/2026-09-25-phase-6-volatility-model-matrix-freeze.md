# Phase 6 Volatility Method Review and Model-Matrix Freeze

**Date:** 2026-09-25

**Parent:** `2026-09-22-phase-6-volatility-forecasting-plan.md`
**Horizon authority:** `2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`

## 1. Gate status and authority

The two H=8 Stage B label bundles and the two aligned canonical feature stores
now pass full source, identity, target, and hash replay. This amendment records
the required paper-method feasibility review and freezes the executable
volatility matrix before any evaluation prediction is inspected.

This document authorizes implementation, CPU smoke tests, manifest bootstrap,
and—only after those checks pass—execution of the frozen matrix. It does not
authorize changing the target, horizon, rows, model list, loss, scaling, or
epoch from evaluation feedback.

## 2. Paper-method feasibility classification

| Method | Actual contract and required inputs | Phase 6 classification |
|---|---|---|
| Li (2024), TimeMixer | Daily-market multiscale model; trailing annualised log-return volatility target; train/validation/test workflow and variables that do not match the saved future raw-change target | Background evidence only. Adding it would change target semantics and introduce an unplanned validation/tuning protocol. |
| Xu et al. (2024), GINN | Daily one-step variance target derived from log returns; LSTM with a GARCH-informed loss and a tuned regularisation weight | Limitation evidence only. The repository's earlier adaptation was scale-incompatible, and an exact identical-row reproduction is not available without retuning. |
| Yihuan et al. (2026), GARCH-BiLSTM | Commodity-market trailing variance with GARCH-family inputs, bidirectional recurrent layers, Huber training, and a different calendar split | Background evidence only. Its target, domain inputs, and split are not an exact match; the accessible method description is insufficient for a strict reproduction on the saved rows. |
| Peter et al. (2026), GARCH/LSTM stacking | Parallel GARCH and LSTM predictions with an interaction feature and a linear meta learner | Principled prediction-market adaptation. The existing repository design already implements the core `[g, l, g*l]` stack; Phase 6 must replace the old target/return logic and retain chronological OOF meta-features. |
| Fang and Ślepaczuk (2026), regime-aware volatility | High-frequency equity-index log realised variance, HARQ and filtered Markov GJR-GARCH regime inputs, expanding walk-forward evaluation | Infeasible as an identical-row comparator with current data. Required regime, realised-kernel/order-book, and index-return inputs are absent. |

Primary method records: [TimeMixer](https://arxiv.org/abs/2410.09062),
[GINN](https://arxiv.org/html/2410.00288),
[Peter et al. DOI](https://doi.org/10.1016/j.array.2026.100700), and
[Fang and Ślepaczuk](https://arxiv.org/abs/2606.09478).

No additional paper architecture joins the mandatory matrix. This is a
pre-evaluation feasibility decision, not a result-based exclusion.

## 3. Frozen target and optimization contract

The scientific target remains:

```text
RV(t,8h) = sum_{j=1..8} (p[t+j] - p[t+j-1])^2
```

The optimization unit is fixed squared probability points:

```text
y_scaled = 10,000 * RV_raw
y_raw_prediction = y_scaled_prediction / 10,000
```

This is a constant unit conversion, not a train-fitted target transform.
Every scientific metric is computed after inversion in raw realised-variance
units.

All neural models use:

- seed `0`;
- batch size `512`;
- Adam with learning rate `1e-4` and weight decay `0`;
- Smooth L1 loss with `beta=1.0` in scaled units;
- gradient-norm clipping at `5.0`;
- one uninterrupted 50-epoch trajectory;
- checkpoints and evaluation snapshots at epochs `5`, `15`, and `50`; and
- epoch 50 as the predeclared principal result.

There is no validation split, early stopping, scheduler, restart selection,
or evaluation-driven checkpoint choice.

## 4. Frozen neural architectures

Every neural prediction head is:

```text
Linear(input_width,128) -> GELU
-> Linear(128,64) -> GELU
-> Linear(64,1) -> Softplus(beta=1, threshold=20)
```

Softplus makes the saved scaled and raw predictions nonnegative by
construction.

The raw learned references retain the already declared Phase 5 encoders:

- Raw-OHLCV MLP: flattened `[64,5]`, hidden widths
  `[512,512,256,256,128]`, ReLU, dropout `0.1`;
- Raw-OHLCV LSTM: three recurrent layers with hidden widths `[50,30,20]` and
  dropout `[0.2,0.1,0.05]` between layers.

The canonical framework and all temporal configurations use train-only
coordinate standardisation with population standard deviation, a `1e-8`
floor, and clipping to `[-10,10]`. Raw baselines consume the exact saved
model-ready `[64,5]` contexts without a separately fitted row filter.

## 5. Frozen current-round learned matrix

The representation matrix remains the 11 configurations precommitted by the
temporal-encoder plan:

```text
H0, HC-SL, HC-ST, HB-SL, HB-ST,
HC-AL, HC-AT, HB-AL, HB-AT, HC-DC, HB-DC
```

Each is fitted independently for both walks on the exact saved volatility
rows: 22 framework trajectories. Raw MLP and Raw LSTM add four comparator
trajectories, producing 26 active trajectories in total. The non-learned
references below are evaluated on the same ordered rows but are not training
trajectories.

By user-directed scope amendment on 2026-09-25, adapted GARCH--LSTM is moved
to the next experiment round. It is not an active matrix entry, mandatory
pending entry, or current-round completion condition. The following safeguards
are retained only as requirements for reconsidering it later:

- the econometric branch models raw probability changes, not log returns;
- its forecast is the sum of the eight recursively forecast conditional
  one-hour variances;
- all location, scale, GARCH parameters, caps, and fallbacks are fitted from
  permitted training history only;
- training meta-features are five-fold contract-aware expanding OOF
  predictions from both the GARCH and Raw-LSTM branches;
- meta-features are `[garch, lstm, garch*lstm]`;
- the meta-feature scaler is train-OOF-only `RobustScaler`;
- the meta learner is ElasticNet with `alpha=1e-4`, `l1_ratio=0.5`, cyclic
  updates, and at most 10,000 iterations; and
- final outputs are clipped at zero, with clipping frequency reported.

The stack is a complete-system comparator. Its internal GARCH or LSTM branch
is not promoted as a separately selected result.

## 6. Frozen references and metrics

Every evaluation table includes:

1. exact zero;
2. the walk-training median target, repeated over evaluation rows; and
3. historical persistence: the sum of the eight squared raw close changes in
   the immediately preceding context interval.

Persistence uses only the saved historical context. It remains available when
that context contains the accepted isolated one-hour fill; an
observed-history-only subset is also reported.

Primary raw-unit metrics are MAE, RMSE, MSE, Pearson correlation, and Spearman
correlation. Reports also retain per-contract metrics, positive-target rows,
zero-target rows, context-imputation exposure, starting-price/lifecycle/
activity strata, future-update-count strata marked retrospective, and tail
error concentration. Overlapping-row results remain descriptive unless a
separate predeclared contract/calendar-block interval procedure is added.

## 7. Artifact roots and remaining execution gate

Canonical H0 volatility outputs belong under:

```text
experiments/phase6/volatility_prediction/
  data_preparation/walk{1,2}/
  features/walk{1,2}/
  manifests/
  runs/{framework_h0,raw_ohlcv_mlp,raw_lstm}/walk{1,2}/seed0/
  reports/
```

Temporal configuration outputs remain under the sibling plan's
`experiments/phase6/encoder_variants/` root.

CUDA execution remains blocked until the implementation passes CPU forward,
loss/backward, nonnegative-output, occupied-path, provenance-failure, and
identical-row smoke tests and a manifest-only bootstrap freezes every expected
trajectory.
