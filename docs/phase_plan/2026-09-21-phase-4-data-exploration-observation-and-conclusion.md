# Phase 4 Data Exploration: Observation and Conclusion

**Date:** 2026-09-21
**Status:** Phase 4 data-selection and exploratory-analysis stage concluded;
no Phase 4 encoder, downstream head, or baseline was trained
**Next stage:** Phase 5 walk-forward implementation, encoder retraining,
downstream-task redefinition/training, and matched baseline comparison

## 1. Scope and phase boundary

Phase 4 investigated the data and evaluation contract required after Phase 3
showed that a single final lifecycle tail and absolute next-close prediction
were poor primary tests of representation quality. The work covered:

- global-calendar rather than per-contract-relative evaluation;
- lifecycle staleness and causal recent-activity selection in the pre-December
  2025 archive;
- collection and source auditing of a recent FinData cohort over
  `[2025-12-01, 2026-09-01)` UTC;
- native 15-minute and one-hour movement, gap, staleness, pruning, and
  forward-fill sensitivities; and
- selection of the recent one-hour series for the next exploratory training
  loop.

This phase produced data evidence and a handoff decision, not model-performance
results. The walk-forward builder, fold-specific encoder training, downstream
probability-movement task, and strict baseline matrix move together to Phase 5.
No Phase 4 analysis artifact may be presented as a trained-model result.

## 2. Recent FinData cohort and source audit

The final raw-first cohort contains 50 retrospectively diverse Polymarket
conditions. It was selected for source diagnosis and recent-regime exploration;
the final Phase 5 universe must still be reconstructed causally at each walk
cutoff.

| Native source | Raw rows | Quarantined | Clean rows | Retained |
|---|---:|---:|---:|---:|
| 15-minute | 325,730 | 116 | 325,614 | 99.9644% |
| one-hour | 107,635 | 147 | 107,488 | 99.8634% |

The forward-confirmed `condition-orientation-v2-forward-confirmed` rule retains
persistent crashes or repricings and quarantines transient complementary
reversions and unsupported wide candles. It found candidates in 10 contracts
at 15 minutes and 12 contracts at one hour. All raw files remain immutable,
prices are never inverted, removed timestamps remain gaps, and every decision
records `quarantine_available_at` because its confirmation uses later bars.

The small removal rates show that the operational anomaly is rare in aggregate
and can be pruned within the predeclared 1% batch budget. They do **not** prove
that every retained condition candle is the YES price: the endpoint omits token
identity, complement changes near 0.5 may be undetectable, and persistent
orientation changes are deliberately retained to avoid deleting genuine
market crashes. The token-identified fallback has direct YES provenance but is
too sparse: 28,359 YES trades from 12 conditions yielded only 448 complete
four-hour `seq64+h2` rows, all from two related World Cup contracts.

Accordingly, the selected recent one-hour condition candles are an explicitly
exploratory training source. An affected-contract exclusion sensitivity and
the raw-versus-clean provenance remain required in Phase 5.

## 3. Observed-only probability dynamics

The following figures use exact native cadence and the clean files after
pruning. No forward-filled endpoint contributes to these primary descriptive
statistics.

| Native source | Horizon | Targets | Exact zero | Non-zero | `abs(delta) > 0.005` | `abs(delta) > 0.01` | Mean absolute move |
|---|---:|---:|---:|---:|---:|---:|---:|
| 15-minute | 15 minutes | 265,672 | 31.53% | 68.47% | 17.32% | 9.63% | 0.002765 |
| 15-minute | 30 minutes | 231,649 | 28.46% | 71.54% | 19.42% | 11.44% | 0.003337 |
| 15-minute | 60 minutes | 192,089 | 25.15% | 74.85% | 22.24% | 13.99% | 0.004316 |
| one-hour | 60 minutes | 99,844 | 29.66% | 70.34% | 19.48% | 11.82% | 0.003583 |
| one-hour | 120 minutes | 94,743 | 26.87% | 73.13% | 21.37% | 13.80% | 0.004432 |

Approximately 68%--75% of valid targets change at all, and 17%--22% move by
more than half a percentage point. The recent cohort is therefore not
predominantly frozen at these horizons. Pruning barely changes ordinary
movement frequency but removes extreme-tail influence: at the native one-hour
horizon, the share above 0.005 changes from 19.68% raw to 19.48% clean while
mean absolute movement falls from 0.005750 to 0.003583. At the 30-minute
horizon, the 15-minute share above 0.005 changes from 19.46% to 19.42% while
mean absolute movement falls from 0.003802 to 0.003337.

## 4. Internal timestamp gaps

Gap coverage is measured only between each contract's first and last observed
timestamp. It does not infer missing time before the first or after the last
row.

| Resolution | Observed rows | Internal missing rows | Coverage | Gap events | Maximum missing run |
|---|---:|---:|---:|---:|---:|
| 15-minute | 325,614 | 186,007 | 63.64% | 59,892 | 1,106 bars = 276.5 hours |
| one-hour | 107,488 | 20,455 | 84.01% | 7,594 | 275 bars = 275 hours |

The longest bounding observation gaps are about 11.5 days. Median
contract-level coverage is 68.33% at 15 minutes and 91.78% at one hour. At 15
minutes, 26/50 contracts fall below 75% coverage and 11/50 below 50%; at one
hour, 12/50 fall below 75% and 6/50 below 50%.

### 4.1 Consecutive-gap distribution

| Resolution | Missing bars per gap | Gap events | Missing rows | Share of missing rows |
|---|---|---:|---:|---:|
| 15m | 1 | 31,881 | 31,881 | 17.14% |
| 15m | 2 | 11,386 | 22,772 | 12.24% |
| 15m | 3 | 5,529 | 16,587 | 8.92% |
| 15m | 4 | 3,074 | 12,296 | 6.61% |
| 15m | 5--8 | 4,782 | 28,944 | 15.56% |
| 15m | 9--16 | 2,095 | 24,013 | 12.91% |
| 15m | 17--32 | 765 | 17,007 | 9.14% |
| 15m | 33--96 | 320 | 16,090 | 8.65% |
| 15m | more than 96 | 60 | 16,417 | 8.83% |
| 1h | 1 | 4,534 | 4,534 | 22.17% |
| 1h | 2 | 1,352 | 2,704 | 13.22% |
| 1h | 3 | 617 | 1,851 | 9.05% |
| 1h | 4 | 319 | 1,276 | 6.24% |
| 1h | 5--8 | 484 | 2,977 | 14.55% |
| 1h | 9--16 | 181 | 2,141 | 10.47% |
| 1h | 17--32 | 65 | 1,435 | 7.02% |
| 1h | 33--96 | 31 | 1,632 | 7.98% |
| 1h | more than 96 | 11 | 1,905 | 9.31% |

Most **events** are short: 86.61% of 15-minute gaps and 89.83% of one-hour
gaps contain at most four missing bars. Short gaps do not contain most missing
**rows**, however: complete gaps of at most four bars contain only 44.91% of
the missing 15-minute rows and 50.67% of the missing hourly rows. A long gap is
one event but may contribute hundreds of rows. The audit's bounded-fill counts
are all-or-nothing: a complete gap is filled only when its total length is at
or below the cap; the first few rows of a longer gap are not partially filled.

Missingness is distributed but concentrated toward the sparsest contracts.
The 10 contracts with the most missing rows contribute 57.0% of the 15-minute
total and 60.7% of the hourly total; the top 20 contribute 87.2% and 90.1%.
Nevertheless, 47/50 contracts have at least one 15-minute gap longer than four
bars and 28/50 have at least one hourly gap longer than four bars, so long gaps
are not isolated to only one or two contracts.

### 4.2 Pruning and cross-resolution evidence

Pruning is not the source of the gap problem.

| Resolution | Raw missing | Clean missing | Added by pruning | Coverage change |
|---|---:|---:|---:|---:|
| 15-minute | 185,891 | 186,007 | 116 | -0.023 percentage point |
| one-hour | 20,308 | 20,455 | 147 | -0.115 percentage point |

The two native endpoints also return different sparse views. Of 20,455
missing native-hour slots, 6,232 (30.47%) contain at least one native
15-minute candle and 107 (0.52%) contain all four. Conversely,
113,390/186,007 (60.96%) missing 15-minute slots fall inside an hour for which
the native one-hour endpoint returned a candle. Missing rows therefore cannot
automatically be interpreted as verified no-trade intervals.

## 5. Forward-fill sensitivities

Forward filling was simulated in memory; the saved raw and clean Parquet files
remain irregular and unchanged. Every synthetic row uses the prior observed
close for `open=high=low=close`, sets `volume=0`, and must retain
`is_imputed`, original gap length, and time since the last observation. No
random or volatility-noise augmentation is used.

### 5.1 Grid coverage

| Resolution and cap | Filled rows | Remaining missing | Post-fill coverage | Synthetic share of retained rows |
|---|---:|---:|---:|---:|
| 15m, one-bar gaps | 31,881 | 154,126 | 69.88% | 8.92% |
| 15m, gaps up to four bars | 83,536 | 102,471 | 79.97% | 20.42% |
| 1h, one-bar gaps | 4,534 | 15,921 | 87.56% | 4.05% |
| 1h, gaps up to four bars | 10,365 | 10,090 | 92.11% | 8.79% |

Unbounded filling is rejected: it would make 36.36% of the completed
15-minute grid and 15.99% of the completed one-hour grid synthetic, including
multi-day stale blocks.

### 5.2 One-hour staleness after filling

The selected one-hour sensitivity fills only isolated one-bar gaps.

| Horizon | Mode | Targets | Exact-zero count | Exact-zero rate | Targets touching a fill |
|---|---|---:|---:|---:|---:|
| 1h | observed only | 99,844 | 29,612 | 29.66% | 0% |
| 1h | one-bar filled grid | 108,912 | 35,705 | 32.78% | 8.33% |
| 1h | fill-touched subset | 9,068 | 6,093 | 67.19% | 100% |
| 2h | observed only | 94,743 | 25,454 | 26.87% | 0% |
| 2h | one-bar filled grid | 106,741 | 29,621 | 27.75% | 11.24% |
| 2h | fill-touched subset | 11,998 | 4,167 | 34.73% | 100% |

At the direct one-hour horizon, filling raises pooled exact-zero movement by
3.12 percentage points. At two hours it raises the rate by 0.88 percentage
point. The share moving by more than 0.005 changes from 19.48% to 18.81% at
one hour. The aggressive four-bar hourly sensitivity raises post-fill coverage
to 92.11%, but raises one-hour zero movement to 36.08% and makes 14.69% of
one-hour targets touch a synthetic row. It is not selected.

### 5.3 Fifteen-minute staleness after filling

The four-bar 15-minute sensitivity covers the same maximum clock duration as
one missing hourly bar, but is substantially more intrusive.

| Horizon | Observed targets | Filled-grid targets | Zero before | Zero after | Targets touching fills |
|---|---:|---:|---:|---:|---:|
| 15m | 265,672 | 401,078 | 31.53% | 46.13% | 33.76% |
| 30m | 231,649 | 395,189 | 28.46% | 36.19% | 41.38% |
| 60m | 192,089 | 384,689 | 25.15% | 29.69% | 50.07% |

Within the 135,406 direct 15-minute targets touching a fill, 74.76% have zero
movement. The share moving by more than 0.005 falls from 17.32% to 13.56% at
15 minutes, from 19.42% to 16.68% at 30 minutes, and from 22.24% to 19.30% at
60 minutes. A conservative one-bar 15-minute fill produces zero rates of
38.37%, 29.76%, and 26.96% at the respective horizons and remains the only
filling rule allowed by the frozen exploratory contract.

Forward filling therefore repairs context continuity; it does not create new
observed market information. Filled-grid target counts and zero rates are
diagnostics, not additional ground-truth labels.

## 6. Qualitative conclusion for data before December 2025

The earlier 2024--2025 archive remains useful evidence, but its main lesson is
evaluation design rather than a preferred new training source:

- timestamps and aggregate one-hour/four-hour sample capacity are adequate;
- prediction-market activity declines strongly late in a contract's lifecycle,
  and inactive terminal tails can dominate a final chronological holdout;
- one-hour and four-hour eight-hour targets have nearly identical staleness,
  lifecycle drift, and movement distributions after duration matching, so the
  extra hourly rows are highly overlapping rather than four times as much
  independent information;
- a causal recent-activity rule is more defensible than retrospective
  last-change truncation and should be shared by the framework and baselines;
- absolute next-close prediction is dominated by copying the current close and
  is not an informative primary representation probe; and
- per-contract lifecycle fractions can permit cross-contract calendar
  lookahead in a pooled model, so global-calendar walk-forward evaluation is
  required.

These findings motivate the next-stage walk-forward protocol and continuous
signed probability-movement target. They do not establish a clean causal claim
that every post-December-2025 contract is more dynamic, because the provider,
selection rule, market mix, and missingness mechanisms differ.

## 7. Phase 4 decision

Phase 4 closes with the following decision:

1. Use the recent clean native one-hour FinData series as the primary
   **exploratory** source for the next training loop.
2. Permit only complete isolated one-hour gaps to be forward-filled causally.
   Filled rows may provide historical context but may not be decision or target
   endpoints. Longer gaps split sequences.
3. Keep observed-only movement as the primary label/evaluation population and
   carry explicit imputation metadata in model inputs.
4. Retain native 15-minute data as a resolution sensitivity, not the primary
   source; four-bar filling is diagnostic only because it materially inflates
   synthetic exposure and zero movement.
5. Use global-calendar walk-forward train/test intervals. Universe selection,
   quarantine availability, imputation, scaling, activity eligibility, window
   construction, and target maturity must be causal inside every walk.
6. Train separate encoder, downstream-head, and baseline weights per walk;
   later-walk history must never revise an earlier-walk model or prediction.
7. Use continuous signed probability movement as the primary downstream
   regression target and exact zero movement as the mandatory reference.
8. Preserve the condition-candle token-identity limitation, quarantine rates,
   and affected-contract exclusion sensitivity in every result claim.

## 8. Phase 5 handoff and execution gate

Phase 5 owns the next complete experimental loop:

1. freeze revised recent-period global-calendar walk boundaries and a
   cutoff-local universe rule;
2. implement the one-hour walk builder, causal one-bar filling, activity mask,
   observed-endpoint target eligibility, and replayable manifests under
   `scripts_v2/`;
3. validate contract identities, candle availability, quarantine availability,
   target maturity, gap handling, and cross-walk isolation before training;
4. predeclare the encoder, downstream-task, and matched-baseline matrix;
5. train fold-specific encoders and redefined downstream heads at fixed budgets
   without validation-based early stopping or test-driven selection; and
6. replay predictions and report per-walk, pooled, contract-macro, lifecycle,
   source-sensitivity, and imputation-exposure results on identical rows.

No Phase 5 encoder, downstream, or baseline training is authorized until items
1--4 are implemented and validated. The earlier four-hour top-80 Phase 4
contract remains historical feasibility evidence; it was not executed and is
superseded as the next-stage primary source by this conclusion.

### 8.1 Post-conclusion Phase 5 capacity check

A subsequent no-training capacity audit tested the selected hourly rules under
a balanced two-walk candidate. With a 64-hour context and one-hour target, the
two folds retain 31,828/37,173 active training rows and 21,401/10,086 active
supported evaluation rows over 36/22 training and 14/16 evaluation contracts.
After excluding every quarantine-affected contract, training retains
19,903/26,927 rows and evaluation retains 15,622/7,445 rows over 10/11
contracts. A 256-hour context is materially weaker, and five monthly walks
produce a highly concentrated final fold. This supports `seq64` and the
balanced two-walk schedule as implementation candidates only. It does not
clear the gate because the 50-contract cohort was retrospectively selected and
evaluation must replay quarantine availability causally. See
`docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md`.

## 9. Evidence and reproduction

Primary detailed evidence:

- `docs/data_analysis/2026-09-20-findata-expanded-recent-cohort.md`
- `docs/data_analysis/2026-09-21-findata-forward-confirmed-quarantine.md`
- `docs/data_analysis/2026-09-21-findata-native-15m-1h-dynamics.md`
- `docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md`
- `docs/data_analysis/2026-09-20-phase4-top80-1h-timestamp-capacity.md`
- `docs/data_analysis/2026-09-20-phase4-top80-1h-activity-suitability.md`
- `docs/data_analysis/2026-09-20-phase4-top50-top80-1h-4h-comparison.md`
- `docs/phase_plan/2026-09-20-phase-4-data-selection-and-walk-forward-contract.md`

Reproduce the recent native-frequency results with:

```bash
.venv/bin/python3 scripts_v2/analyze_findata_native_dynamics.py \
  --maximum-fill-bars 1 --overwrite

.venv/bin/python3 scripts_v2/analyze_findata_native_dynamics.py \
  --maximum-fill-bars 4 \
  --output-dir data_new/findata/polymarket/historical_diverse_top50_2025-12-01_2026-08-31/analysis/native_15m_1h_dynamics_cap4 \
  --overwrite

.venv/bin/python3 scripts_v2/analyze_findata_native_gaps.py --overwrite

.venv/bin/python3 scripts_v2/analyze_findata_walk_capacity.py --overwrite
```
