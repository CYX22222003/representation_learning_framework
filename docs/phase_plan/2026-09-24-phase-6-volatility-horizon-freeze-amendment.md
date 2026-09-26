# Phase 6 Volatility Horizon-Freeze Amendment

**Date:** 2026-09-24  
**Status:** Approved; primary horizon frozen before label construction  
**Parent authority:** `2026-09-22-phase-6-volatility-forecasting-plan.md`

## 1. Decision

The primary Phase 6 volatility forecast horizon is frozen to:

```text
native interval delta = 1 hour
primary horizon H = 8 hours
future target interval = (t, t + 8h]
formula version = phase6-future-realised-variance-raw-probability-v1
RV(t,8h) = sum_{j=1..8} (p[t+j] - p[t+j-1])^2
```

The target uses eight squared raw probability changes. It is not square-rooted,
averaged, annualised, or constructed from percentage/log returns.

This decision was made from the predeclared training-period audit of
`H={2,4,8,24}`. Evaluation-period outcomes and all model performance were
excluded. Evaluation information was limited to row and contract capacity.

## 2. Evidence used for the freeze

| Walk | Training rows | Training contracts | Evaluation-capacity rows | Evaluation-capacity contracts | Training zero rate | Largest interval share |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32,470 | 46 | 27,806 | 23 | 0.6591% | 1.8602% |
| 2 | 53,112 | 43 | 12,115 | 25 | 0.4330% | 2.6334% |

Additional training-only diagnostics relevant to the decision are:

| Walk | RV lag-1 Pearson | Historical/future RV Spearman | Mean non-zero future updates |
|---:|---:|---:|---:|
| 1 | 0.7167 | 0.8602 | 6.0324 |
| 2 | 0.9326 | 0.7916 | 5.9668 |

Eight hours is selected because it:

1. contains several future increments and is interpretable against the fixed
   64-hour context;
2. reduces the training zero mass below 0.7% in both walks;
3. retains adequate row and contract capacity in both walks;
4. avoids the larger row/contract loss of 24 hours; and
5. has less extreme overlap-driven label persistence than 24 hours.

The strong historical/future relationship does not invalidate the target. It
requires historical realised-variance persistence to remain a mandatory
reference. The high stride-one dependence also requires descriptive reporting
or predeclared contract/calendar-block uncertainty rather than IID row-wise
inference.

## 3. Rules carried into the label bundle

The replacement bundle must preserve every rule in the parent plan:

- the historical context remains the 64 native one-hour bars ending at the
  observed forecast origin;
- all eight future closes exist at exact hourly timestamps in the same
  contract and uninterrupted gap segment;
- every future target candle is observed; an imputed target candle invalidates
  the row;
- context imputation remains permitted only under the accepted isolated-one-
  bar Phase 5 rule and is stored explicitly;
- training target availability is strictly before the walk cutoff;
- evaluation target availability is no later than the evaluation end;
- evaluation contracts retain the common training-support rule; and
- framework models and every baseline consume identical saved rows and target
  values.

Retrospective `end_date_search` metadata remains reporting-only because its
midnight timestamp can denote the start of a valid final market day. Contract
and source eligibility instead use exact same-condition timestamps, actual
source boundaries, and gap segments.

## 4. Scope opened and still blocked

This amendment completes ordered gate 3 of the parent plan and authorizes only
the next gate: build and independently validate the two walk-specific H=8
label bundles.

It does not authorize model training. Frozen-feature alignment, literature-
method feasibility review, complete model-matrix freeze, CPU smoke tests, and
row-identity replay remain required before any Phase 6 CUDA execution.

The 2-, 4-, and 24-hour audit results remain data-characterisation evidence.
They are not model-selection alternatives or automatic sensitivity runs.

## 5. Evidence artifacts

The decision is tied to the hash-validated Stage A artifacts under:

```text
experiments/phase6/volatility_prediction/data_exploration/
```

The machine-readable freeze record is:

```text
experiments/phase6/volatility_prediction/manifests/horizon_freeze_h8.json
```

The Stage A manifest remains unchanged and correctly records that the audit
itself did not freeze a horizon. This amendment and the separate freeze record
are the post-audit approval evidence.
