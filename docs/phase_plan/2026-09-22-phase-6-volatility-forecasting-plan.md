# Phase 6 Volatility Forecasting Task Plan

**Date:** 2026-09-22

**Status (updated 2026-09-26):** Gates 1--8 are complete and gate 9 is partial.
The Stage A audit
and H=8 decision remain frozen. Both replacement label bundles replay every
target from source closes, and both canonical 445-dimensional feature stores
are identity-aligned and byte-identical to their frozen Phase 5 sources. The
paper feasibility decision and 26-entry current-round volatility matrix are
frozen by `2026-09-25-phase-6-volatility-model-matrix-freeze.md`. CPU smoke
tests and manifest-only bootstrap pass. H0, Raw-OHLCV MLP, and Raw LSTM are
complete for both walks at epochs 5/15/50 with CPU prediction replay. The
ten temporal/control configurations per walk remain pending, so the matrix
and task are not complete. By user-directed scope amendment on 2026-09-25,
GARCH--LSTM is deferred to a later round and is not an active or mandatory
current-round entry.

**Predecessor:** `2026-09-21-phase-5-experiment-plan.md`

**Horizon amendment:** `2026-09-24-phase-6-volatility-horizon-freeze-amendment.md`

**Scope:** Phase 6 Task 1 only. The temporal-encoder-alternative study is a
separate Phase 6 task specified in
`2026-09-22-phase-6-temporal-encoder-variants-plan.md`. All 11 frozen encoder
configurations will evaluate the new volatility task only after this plan's
independent horizon and label gates pass.

## 1. Purpose and authority

This document specifies the first Phase 6 task: redefine volatility
forecasting as a genuine future-interval prediction problem, examine whether
the selected recent prediction-market data can support that task, prepare a
replayable target bundle, and only then execute a matched model comparison.

For this task, this document supersedes the historical volatility-label
contract in:

```text
data/task_labels/volatility_prediction/rv_4h_seq64_top50.npz
```

That bundle remains reproducible Phase 1/2 characterisation evidence. It must
not be reused for Phase 6 claims because its target window is a one-step-shifted
copy of the input window: 63 of 64 prices and 62 of 63 returns are already in
the input.

This plan does not authorize immediate training. Data exploration, horizon
selection, label construction, row-identity validation, model-matrix freezing,
and smoke tests must pass in that order before CUDA execution begins.

## 2. Research question

At an observed decision time `t`, can information available no later than `t`
predict the amount of price variation that will be realised across a strictly
future interval `(t, t + H]`?

This is an interval-level volatility-forecasting problem. It is not:

- prediction of the next closing price;
- prediction of a volatility value attached only to the next close;
- a shifted trailing-window volatility nowcast; or
- prediction of signed price direction.

The task is relevant because volatility is a potential input to risk,
liquidity, opportunity-screening, and trading decisions, and because
volatility forecasting is an established downstream task in econometric and
deep-learning market research. The Phase 6 claim, however, is predictive-task
performance on the declared prediction-market data. It is not a claim of a
profitable strategy or a new estimator of volatility.

## 3. Literature-grounded target definition

### 3.1 Primary scientific target

Let:

- `p_t` be the observed prediction-market close at the forecast origin;
- `delta` be the native sampling interval;
- `H` be a forecast horizon satisfying `H > delta`; and
- `H / delta` be an integer.

The symbol `t` denotes the information-availability time of the close, not
blindly the raw candle timestamp. FinData candle timestamps are treated as bar
starts in the existing project contract, so implementation must preserve both
the source bar-start timestamp and the time at which its close is available.
The target interval begins only after that availability time.

Define the contract-local raw probability change in future subinterval `j` as:

$$
r^{PM}_{t+j\delta,\delta}
=
p_{t+j\delta}-p_{t+(j-1)\delta}.
$$

The Phase 6 target is:

$$
\boxed{
RV^{PM}_{t,H,\delta}
=
\sum_{j=1}^{H/\delta}
\left(
p_{t+j\delta}-p_{t+(j-1)\delta}
\right)^2
}
$$

The target uses only price changes realised after the forecast origin. The
input and target may share the boundary price `p_t`, which is known at the
decision time, but they share no close-to-close return.

For example, with native hourly data and an eight-hour horizon:

$$
RV^{PM}_{t,8h,1h}
=
(p_{t+1}-p_t)^2
+(p_{t+2}-p_{t+1})^2
+\cdots+
(p_{t+8}-p_{t+7})^2.
$$

### 3.2 Terminology

The formula above is mathematically a **realised variance** because it sums
squared changes without taking a square root. Some cited literature uses
“realised volatility” as an umbrella term for this quantity. Phase 6 may use
the phrase *realised-volatility forecasting task* for continuity with that
literature, but every artifact and report must identify the numerical target
as future realised variance and display the formula.

The primary label does not:

- take `sqrt(RV)`;
- divide by `H / delta`;
- annualise the result;
- subtract a future sample mean; or
- use percentage or log returns.

Any later square-root or log-target experiment would be a separately named
modelling sensitivity. It cannot silently replace the primary scientific
target after evaluation results are observed.

### 3.3 Basis in the literature

- Andersen et al., *Modeling and Forecasting Realized Volatility*, supplies
  the sum-of-return-outer-products construction. In the univariate case it
  reduces to the sum of squared intraperiod returns used above.
- Restocchi et al. (2019), *The Stylized Facts of Prediction Markets: Analysis
  of Price Changes*, supports representing prediction-market movement as raw
  bounded price changes rather than conventional percentage returns.
- Xi et al. (2026), *Volatility in Prediction Markets: A Structural Approach*,
  motivates important domain limitations: conditional variation may depend on
  the probability level, time to resolution, whether an update occurs,
  activity, spread, and liquidity. It does not replace the primary realised-
  variance label.

The Phase 6 target is therefore an adaptation of an established realised-
variance estimator to raw prediction-market probability changes. A novel
volatility definition is not a Phase 6 objective.

## 4. Source data and controlled starting point

The initial data source is the accepted Phase 5 clean native one-hour FinData
condition-candle data from the two independently acquired top-50 walk cohorts.
The same global calendar intervals remain the starting evaluation design:

| Walk | Permitted training interval | Out-of-future evaluation interval |
|---:|---|---|
| 1 | `[2025-12-02, 2026-04-01)` | `[2026-04-01, 2026-06-16)` |
| 2 | `[2026-02-16, 2026-06-16)` | `[2026-06-16, 2026-09-01)` |

The Phase 5 research assumptions remain visible:

1. the two cohorts were retrospectively selected;
2. the approved forward-confirmed pruning is treated as correct offline
   cleaning and is not replayed by quarantine-availability time; and
3. condition-level candles retain a token-orientation limitation even after
   the audited cleaning process.

The starting input context remains 64 one-hour OHLCV bars. This task plan does
not introduce a resolution comparison, a sequence-length comparison, a gated
fusion comparison, or a decoder-variant comparison. Holding those choices
fixed isolates the consequences of correcting the volatility target.

The existing Phase 5 walk-specific epoch-50 VAE, contrastive-CNN, and BYOL-CNN
encoders are the canonical frozen feature sources for the first framework
comparison. They were trained without volatility labels and under the same
walk cutoffs. Reuse is allowed only after source hashes, context identities,
and target-free encoder eligibility replay. A volatility target must never
change which rows trained an encoder. Volatility-eligible contexts absent from
the existing downstream feature stores may be freshly extracted through the
same frozen checkpoint, as in the Phase 5 eight-hour add-on procedure.

## 5. Future-interval label and eligibility contract

For a decision row at `t`, the context is historical and the volatility target
is forward-looking:

```text
historical context                     future target interval
[t - 63h, ..., t]                      (t, ..., t + H]
                    forecast origin ───►
```

A supervised row is eligible only when all of the following hold:

1. `p_t` is an observed close and is the final observed decision endpoint.
2. Every required close `p_{t+delta}, ..., p_{t+H}` exists at the exact native
   timestamp.
3. Every close used by the target is observed. An imputed candle anywhere in
   `(t, t + H]` invalidates the row.
4. The decision and all future target observations belong to one contract and
   one uninterrupted post-quarantine gap segment.
5. The target does not cross a walk cutoff, evaluation end, contract boundary,
   or source-file boundary.
6. A training target is mature strictly before the corresponding walk cutoff.
   An evaluation target is mature no later than the declared evaluation end.
7. Context eligibility and any causal activity rule use information available
   at or before `t`; they do not inspect whether future movement occurs.
8. All framework and baseline models consume the exact same saved decision-row
   identities and target values.

The Phase 5 policy permitting one isolated flat, zero-volume imputed bar may
remain available for the historical input context, with explicit exposure
metadata and an observed decision endpoint. It does not extend to the future
target interval.

Observed future intervals with no price change have `RV = 0` and remain in the
primary task. An “active-update-only” analysis may be reported as a descriptive
stratum, but future activity must not be used to filter the primary population.
Doing so would condition sample eligibility on the label interval and would
define a different forecasting problem.

Stride-one decision rows may have overlapping future intervals. This is not
input-target leakage, but it creates serially dependent labels and prediction
errors. Reports must not treat row-wise observations as independent. Any
confidence interval must use a predeclared contract/calendar-block procedure;
otherwise results remain descriptive.

## 6. Stage A — data exploration before target freezing

No model training is allowed in Stage A. The audit must operate from the clean
walk sources and must separate information used for design from the locked
evaluation outcomes.

### 6.1 Horizon audit

The following procedure governed the formerly unresolved forecast horizon
`H`. Before inspecting model performance:

1. predeclare a small candidate set with `H > 1h`;
2. compute training-period target diagnostics for every candidate;
3. inspect evaluation-period row and contract availability only for basic
   feasibility, without using evaluation target distributions to choose `H`;
4. select one primary horizon using the criteria below; and
5. record the choice in a dated horizon-freeze amendment before labels are
   exposed to model runners.

The primary horizon must:

- contain multiple future increments rather than degenerate to the squared
  next change;
- retain adequate rows and distinct contracts in both walks;
- have a non-degenerate training distribution;
- be interpretable in the one-hour prediction-market setting; and
- be fixed without reference to framework or baseline evaluation performance.

Other audited horizons remain data-characterisation evidence unless a later
document explicitly approves a sensitivity experiment.

**Gate result (2026-09-24):** Steps 1--5 are complete and `H=8h` is frozen by
the dated horizon amendment. Stage B label construction may proceed; model
training remains blocked by the later gates in Section 10.

### 6.2 Required target diagnostics

For each candidate horizon, report at minimum:

- eligible training rows and evaluation-row availability by walk and contract;
- dropped rows by missing intermediate candle, imputed target candle, gap,
  cutoff maturity, evaluation-end maturity, and unsupported contract;
- zero-target rate, positive-target rate, mean, standard deviation, quantiles,
  skewness, and maximum on training rows;
- the share of total realised variance contributed by the largest target
  interval and by the largest squared increment within each target;
- autocorrelation of raw probability changes, squared changes, and the
  resulting realised-variance series;
- relationship between trailing historical realised variance and future
  realised variance;
- target distributions by starting-price band, causal activity state,
  lifecycle reporting bucket, and time-to-resolution bucket when reliable
  resolution metadata is available;
- number of non-zero future updates inside each target interval; and
- historical-context imputation exposure.

The audit must make clear whether apparent variation is broad-based or driven
by a small number of jumps, contracts, timestamps, or near-resolution events.
No outlier may be removed merely because it is large. Any source-error removal
must trace to the existing quarantine evidence or a new documented raw-data
audit independent of model performance.

### 6.3 Stage A deliverables

Stage A ends with:

```text
experiments/phase6/volatility_prediction/data_exploration/
  horizon_capacity.csv
  target_distribution.csv
  target_dependence.csv
  target_strata.csv
  plots/
  report.md
  manifest.json
```

The manifest must contain source paths and hashes, walk boundaries, candidate
horizons, formula version, timestamp interpretation, target-observation rules,
and code revision.

## 7. Stage B — freeze and prepare the supervised task

After the horizon amendment is approved, build one replayable volatility
bundle per walk. The bundle must store at least:

- raw and model-ready historical OHLCV contexts;
- raw future realised-variance targets;
- decision timestamp, target start, target end, and target availability time;
- contract ID, source identity, gap-segment identity, and lifecycle metadata;
- all component future closes or a replayable source-row map;
- the component squared changes and their sum;
- context and target observation/imputation flags;
- training/evaluation membership;
- aligned framework/baseline row indices; and
- source, configuration, row-identity, target-array, and artifact hashes.

Validation must recompute every sampled target from source closes and prove:

```text
stored_RV == sum(stored_component_squared_changes)
```

within a declared floating-point tolerance. It must also prove that no target
return appears in the input context, no target contains an imputed row, no
target crosses a boundary, training targets mature before the cutoff, and all
comparators receive identical rows.

The historical `rv_4h_seq64_top50.npz` path must not be overwritten. New
artifacts belong under:

```text
experiments/phase6/volatility_prediction/data_preparation/
```

## 8. Stage C — prediction and comparison contract

### 8.1 Primary framework

The first framework run uses the canonical five-branch, 445-dimensional concat
representation:

```text
statistical + transformed + VAE + contrastive CNN + BYOL CNN
```

The walk-specific frozen Phase 5 encoders remain fixed. Only the volatility
head is fitted on volatility-eligible training rows. The output must be
nonnegative by construction. The exact head, loss, fixed unit conversion, and
epoch budget must be frozen after Stage A and before any evaluation prediction
is inspected.

The primary scientific target remains raw realised variance regardless of an
invertible optimization-scale conversion. Any train-fitted target transform
must be fitted from the current walk's training targets only, inverted before
scientific metrics, and named explicitly. A log-target model is secondary and
is not automatically authorized by this plan.

### 8.2 Mandatory non-learned references

At minimum, compare against:

1. **Zero variance:** predict `0` for every row.
2. **Training-location reference:** predict a training-only constant such as
   the median, with the statistic frozen per walk.
3. **Historical-volatility persistence:** compute realised variance over the
   immediately preceding observed interval of the same duration `H` and use it
   as the forecast for `(t, t + H]`.

The historical-persistence reference must obey the same decision-time and
data rules as the learned models. On a context containing the permitted
isolated one-hour fill, the reference may use that same causally filled context
so it remains available on identical rows; its results must then be stratified
by context-imputation exposure, with an observed-history-only result reported
separately. It may not read future activity or future target availability
beyond the shared eligibility map.

### 8.3 Learned comparators

The initial strict comparison should contain:

| Model | Role |
|---|---|
| Raw-OHLCV MLP | Minimum learned raw-input reference |
| Raw-OHLCV LSTM | Direct temporal neural benchmark |
| Canonical five-branch framework | Frozen-representation transfer test |

After the target and volatility-training contract pass their independent
gates, the canonical framework row expands into the complete 11-configuration
encoder matrix from the sibling temporal plan. This adds four fixed-width
substitutions, four heterogeneous additions, and two duplicate-width controls
alongside `H0`, for 22 configuration-walk volatility trajectories. These runs
are mandatory even if a variant performed poorly on classification or future
price; the earlier tasks cannot screen the volatility matrix.

All learned models must be refitted independently per walk on identical
volatility rows. Existing Phase 1/2 volatility checkpoints and predictions are
not reusable evidence because both their source pipeline and target differ.

The adapted GARCH--LSTM stack is deferred to the next experiment round. It is
not an active matrix member and cannot block completion of the current round.
Before any later reuse, its GARCH state, scaling, caps, and forecasts must be
audited against permitted historical changes; training meta-features must be
chronological out-of-fold predictions, and test predictions must use only the
walk's training history. Any later stack remains a complete-system comparator,
not evidence about standalone GARCH superiority.

GINN remains limitation evidence unless a later amendment defines a corrected,
scale-compatible adaptation. It is not a mandatory headline comparator for
this first Phase 6 volatility matrix.

### 8.4 Literature-driven model review before matrix freeze

The executable matrix is not frozen merely by carrying forward existing code.
After the target and data audit, review the exact methodology and
reproducibility requirements of the volatility papers already collected,
including:

- Li (2024), *Volatility Forecasting in Global Financial Markets Using
  TimeMixer*;
- Xu et al. (2024), *GARCH-Informed Neural Networks for Volatility Prediction
  in Financial Markets*;
- Yihuan et al. (2026), *Bridging Econometrics and Deep Learning: A Hybrid
  GARCH-BiLSTM Approach*;
- Peter et al. (2026), *A Stacking Model Integrating GARCH and LSTM with
  Feature Interactions for Time-Series Volatility Prediction*; and
- Fang and Ślepaczuk (2026), *Volatility Forecasting and Return Prediction
  under Market Regimes*.

For each candidate, record:

1. its actual supervised target and horizon;
2. required sampling frequency and input variables;
3. preprocessing and fitted econometric components;
4. architecture, loss, output constraint, and evaluation protocol;
5. whether sufficient implementation detail is available; and
6. whether it can consume the identical Phase 6 rows without future
   information or model-specific sample filtering.

Classify each method as an exact reproduction, a principled prediction-market
adaptation, background evidence only, or infeasible with the available data.
Do not implement a model from its title or high-level description alone.
TimeMixer or another paper-derived architecture may be added only through the
pre-evaluation matrix-freeze amendment. The minimum strict comparison in
Section 8.3 remains required even if no additional paper model is feasible.

This task plan does not add branch ablations, multiple-seed estimation, gated
fusion, or controlled decoder variants to the initial target-validity matrix.
Those require separate scope approval and must not delay establishing whether
the corrected volatility task itself is viable.

## 9. Evaluation and reporting

Report raw realised-variance units after inverting any optimization scaling.
The primary metrics are:

- MAE;
- RMSE and MSE;
- Pearson correlation; and
- Spearman correlation.

Also report error skill relative to the zero and historical-volatility
persistence references. Do not report direction or sign agreement as a
volatility metric.

Required breakdowns are:

- Walk 1 and Walk 2 separately;
- pooled out-of-future predictions only after walk-local inference;
- row-weighted and contract-macro results;
- zero versus positive realised variance;
- starting-price bands;
- lifecycle and time-to-resolution strata when metadata is reliable;
- causal activity and future update-count strata, with the latter clearly
  marked as retrospective analysis rather than a deployable filter;
- historical-context imputation exposure; and
- tail concentration, including how much aggregate squared error comes from
  the largest realised-variance observations.

Every table must place non-learned references beside learned models. A model is
not useful merely because its correlation is positive; it must be interpreted
against historical-volatility persistence and the strongly zero-inflated
prediction-market environment.

Because stride-one target intervals overlap, ordinary row-wise standard errors
and IID significance tests are invalid. If uncertainty intervals are included,
the resampling unit and block construction must be frozen before reporting.

## 10. Ordered implementation and execution gates

Gates 1--8 are complete. Six trajectories are complete under gate 9, and the
complete temporal/control downstream manifest has passed its CPU smoke and
row-identity freeze without training. Its 20 volatility trajectories are the
current-round next actions. GARCH--LSTM is deferred to a later round.

1. **Complete the literature-grounded definition.** The future realised-
   variance formula and raw probability-change convention are now fixed.
2. **Implement the read-only data audit.** Do not construct model predictions.
3. **Review the audit and freeze `H`.** Record a dated amendment and formula
   version.
4. **Build and validate walk-specific label bundles.** Preserve the old bundle.
5. **Align or extract frozen canonical features.** Prove encoder eligibility
   and weights are target-independent.
6. **Complete the paper-method feasibility review.** Classify candidate
   volatility models before deciding whether any joins the mandatory matrix.
7. **Freeze the complete model matrix.** Record architectures, loss, output
   constraint, optimization units, epoch budgets, checkpoints, and artifact
   roots before evaluation.
8. **Run CPU smoke tests and row-identity replay.** Default launch commands
   should not train without an explicit execution flag.
9. **Train walk-specific heads and baselines.** No validation split, early
   stopping, or evaluation-driven configuration selection.
10. **Replay every prediction from saved checkpoints.** Verify source, row,
   scaler, target, and prediction hashes.
11. **Report the complete frozen matrix.** Do not promote a horizon,
    transformation, or model because it looks best on evaluation data.

## 11. Limitations to carry into the final report

1. Realised variance is an ex-post proxy for latent conditional variance; it
   is not the latent process itself.
2. Hourly sampling may miss intrahour variation, while sparse or unchanged
   candles create a substantial mass at zero.
3. The bounded probability level, time to resolution, update occurrence,
   volume, spread, and liquidity may structure prediction-market volatility,
   as emphasized by Xi et al. The available candle data does not contain a
   reliable bid--ask spread series, so the full structural formulation cannot
   be reproduced.
4. Condition-candle token orientation remains an accepted source limitation.
5. Retrospective market selection and offline anomaly cleaning limit external
   validity.
6. Overlapping future intervals reduce the effective number of independent
   observations.
7. A predictive result is not evidence of an exploitable trading strategy
   without fees, liquidity, execution, position, and market-resolution rules.

These limitations support cautious interpretation; they do not require an
unvalidated new volatility definition.

## 12. Exit conditions

The Phase 6 volatility task is complete only when:

- one primary `H` is frozen from the data audit without model-test feedback;
- target capacity and distribution reports exist for both walks;
- every saved target replays from observed future closes;
- no target return overlaps the historical input;
- all framework and baseline rows are identical and hash-verified;
- the collected volatility methods are classified for reproducible and causal
  adaptation before the executable model matrix is frozen;
- the model matrix, output constraint, loss, scaling, and budgets were frozen
  before evaluation;
- all 22 encoder-configuration-by-walk volatility trajectories required by the
  sibling temporal plan exist and use the identical volatility rows;
- every checkpoint and prediction file passes independent replay;
- the complete non-learned and learned comparison is reported per walk and
  pooled; and
- limitations and negative results are documented without replacing the
  target post hoc.

Until these conditions are satisfied, Phase 6 volatility work must be
described as planned, audited, or partially implemented rather than completed.
