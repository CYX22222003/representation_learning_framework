# Raw-to-Sequence Split and Preprocessing Contract

Date: 2026-09-20
Status: Replacement implementation launched; Phase 2 execution remains paused

> **Phase 4 conclusion / Phase 5 transition (2026-09-21):** Phase 3 implemented and validated the
> raw-time-first correction specified here. Its downstream study then exposed
> a separate evaluation-design limitation: one final-20% holdout concentrates
> near-settlement, persistent observations. Phase 5 retains every causality
> requirement in this document but replaces the single boundary with
> predeclared global calendar-time walks. A per-contract lifecycle split alone
> is insufficient for the pooled model because training rows from one contract
> can be later in calendar time than evaluation rows from another. Each walk
> must fit preprocessing and construct windows from only information available
> before its global cutoff. Lifecycle position is a reporting stratum, not the
> primary split. See the Phase 3 conclusion document.
> The earlier four-hour top-80 Phase 4 contract was not executed. Phase 4
> instead concluded by selecting recent clean native one-hour FinData with
> causal isolated-one-bar filling for the next exploratory loop. Revised
> recent-period walks, cutoff-local selection, candle/quarantine availability,
> and replay validation must be frozen in Phase 5. See
> `docs/phase_plan/2026-09-21-phase-4-data-exploration-observation-and-conclusion.md`.

Tracking issue: [#15 — Rebuild split-safe data pipeline before resuming Phase 2](https://github.com/CYX22222003/representation_learning_framework/issues/15)

## Purpose

This document records a Phase-2-wide validity defect discovered in the shared
data pipeline and defines the required replacement contract. Existing raw
artifacts, checkpoints, feature bundles, predictions, and reports must be
preserved for reproducibility. They are historical characterisation evidence,
not leakage-free confirmatory evidence.

## Legacy pipeline and defect

The current implementation in `src/data_processing/data_processing.py` applies
the following order independently to each contract:

```text
complete raw contract
-> forward-fill/interpolate the complete contract
-> fit volume mean/std on the complete contract
-> generate all stride-one length-64 windows
-> split the windows chronologically 80/20
```

This ordering has two consequences:

1. volume normalisation for training rows uses statistics influenced by the
   later test period; and
2. the last training window and first test window share 63 of 64 raw
   observations on the current length-64 setup.

The audit confirmed both properties for all 50 contracts in the active 4-hour
dataset. The first property is direct fitted-preprocessing leakage. The second
means that the documented 80/20 boundary is a boundary between generated
windows, not a clean raw-observation holdout. Whether historical context may be
carried into the first test window is a modelling choice, but it must be
explicit and must not allow test-period observations or fitted statistics to
influence training.

## Required replacement order

The corrected pipeline must predeclare a per-contract raw-time boundary before
any fitted preprocessing or sample construction:

```text
complete raw contract
-> establish chronological raw train/test boundary
-> fit every imputer/scaler on permitted training history only
-> apply the frozen preprocessing rule causally
-> construct train and test windows under an explicit context policy
-> construct task targets independently within contract and split
-> merge eligible rows across contracts
```

The test-context policy must state whether the first test prediction may use
historical observations from immediately before the boundary. If permitted,
those observations are context only: no test-period value may enter a training
input, training target, scaler, imputer, encoder update, or selection rule.

## Phase 5 global-calendar extension

For a pooled cross-contract Phase 5 model, the raw boundary is a global
timestamp, not a separately calculated fraction of each contract's final row
count. At cutoff `T_k`, preprocessing, universe eligibility, encoder training,
task-head training, and baseline fitting may use only information whose
decision-time availability is before `T_k`. Targets used for training must
also have matured before the cutoff. The immediately following calendar
interval is evaluation-only for that walk.

The primary Phase 5 history is a fixed-duration rolling window rather than an
expanding prefix. A contract must have its complete input inside that rolling
window, and supervised targets must mature before the cutoff. The selected
recent source is native one-hour FinData; its raw candle dates label bar starts,
so availability is `date + 1h`. Only complete isolated one-hour gaps may be
filled, decision and target endpoints remain observed, longer gaps split
sequences, and row eligibility uses the causal prior-24h observed-price-change
rule. A capacity audit supports a 64-hour input and balanced two-walk schedule
as implementation candidates, with 256 hours retained as a sensitivity. These
are not training-ready until cutoff-local universe selection and causal
quarantine replay are implemented and the exact bundle is replay-validated.
See `docs/data_analysis/2026-09-21-phase5-findata-walk-capacity.md`.

Each walk has separate preprocessing parameters, encoder checkpoints, feature
bundles, task heads, and prediction identities. Later history may train the
next walk but can never update an earlier-walk model. Contract lifecycle
position may be calculated for stratified reporting only from metadata known
at decision time; otherwise it must be clearly marked as retrospective.

## Required provenance and tests

The replacement processed bundle must record:

- raw boundary timestamp/index for every contract;
- raw indices or timestamps covered by every generated window;
- preprocessing parameters and the exact training prefix used to fit them;
- the train/test context policy;
- source-file hashes and contract ordering; and
- the global calendar cutoff and evaluation interval for every Phase 5 walk;
- decision-time and target-maturity timestamps for every eligible row;
- fold-specific encoder/checkpoint hashes; and
- processed row identities that downstream label bundles can replay.

Tests must fail when:

- a fitted statistic reads any test-period observation;
- a training window or target contains a test-period observation;
- a horizon crosses a contract or split endpoint;
- a training row, fitted parameter, encoder update, or matured target occurs
  at or after its walk's global cutoff;
- a pooled model uses later calendar information from one contract to predict
  an earlier timestamp from another contract;
- reconstruction from the manifest changes row identities; or
- framework and baseline samples differ under a claimed strict comparison.

## Artifact and execution policy

- Do not delete or overwrite existing artifacts. Legacy generated artifacts
  have been moved into `_old` roots; see `LEGACY_ARTIFACTS.md` for the complete
  old-to-new location map.
- Do not continue Phase-2 training or downstream evaluation on the legacy
  processed or feature bundles.
- Rebuild processed data first, then retrain the neural encoders and regenerate
  deterministic/neural feature bundles because they inherit the old inputs.
- Regenerate price, volatility, and classification label/alignment bundles from
  the corrected identities.
- Rerun only the predeclared comparisons needed for the revised Phase-2 scope.

## Implementation status (2026-09-20)

The raw-time-first builder is implemented in
`src/data_processing/data_processing.py` and exposed by
`scripts/prepare_sequences.py`. The selected context policy is
`isolated_test_windows`: training windows end before the raw boundary and test
windows begin at or after it, so the two sample sets share no raw rows. Missing
values are forward-filled causally, with any leading gaps filled from
training-prefix medians; volume scaling is fitted on that same prefix only.

New builds use the `_split_safe.npz` suffix, refuse overwrite by default, and
write contract IDs, window starts/ends, timestamps, source hashes, preprocessing
parameters, and identity/sequence replay hashes. Unit tests cover test-period
perturbation invariance, boundary separation, causal filling, short-contract
rejection, and deterministic replay identities.

This implementation does not lift the execution pause. The full 4-hour top-50
bundle must still be generated and audited, then all inherited checkpoints,
features, and task-label bundles must be rebuilt before Phase 2 resumes.

## Phase 5 implementation status (2026-09-21)

Phase 5 does not reuse the legacy 80/20 artifacts above. Its global-calendar
builder is implemented in `src/data_processing/phase5_walks.py` and exposed by
`scripts_v3/prepare_phase5_data.py` and
`scripts_v3/validate_phase5_data.py`. Outputs live under
`experiments/phase5/data_preparation/`, not `data/phase3/` or `scripts_v2/`.

The builder validates the approved bounded-fill source before scaling, fits
volume normalisation only on unique candles in the walk training interval,
creates separate encoder-training, mature supervised-training, and next-
interval evaluation populations, and permits pre-cutoff historical context for
evaluation. Encoder rows use context and decision information only; future
target existence, gap status, observation status, and maturity are consulted
only for downstream supervised rows. It requires exact contract/segment
continuity, active prior-24h history, observed decision and target endpoints,
target maturity, and a supported training contract for evaluation. It stores raw and scaled OHLCV,
the four imputation metadata fields, shared two-hour regression/classification
labels, source hashes, row-identity hashes, and replay hashes. This completes
the Phase 5 data-preparation portion of the gate; it does not authorize model
training before the independent model lifecycle, matrix, smoke tests, and
prediction replay are complete.
