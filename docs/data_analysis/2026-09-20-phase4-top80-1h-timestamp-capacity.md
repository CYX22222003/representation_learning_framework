# Phase 4 Top-80 One-Hour Timestamp and Capacity Audit

**Date:** 2026-09-20
**Status:** exploratory data audit; no model training or Phase 4 bundle generation

## Question

This audit checks whether the one-hour Polymarket files contain usable calendar
timestamps and whether they provide enough observations for the proposed Phase
4 encoder and downstream walk-forward experiments.

The comparison uses the same 80 contract identities as the corrected four-hour
lifecycle analysis. This keeps the timeframe comparison interpretable, but it
does not turn the retrospectively chosen top-80 cohort into a deployment-safe
universe. Phase 4 still needs a cutoff-local contract-selection rule.

## Timestamp finding

The raw one-hour feather files do contain a `date` column. It is stored as
`datetime64[ns]`. Across the 80 audited contracts:

| check | result |
|---|---:|
| Raw rows | 192,658 |
| Rows per contract, minimum / median / maximum | 1,286 / 1,649 / 8,344 |
| Earliest / latest timestamp | 2024-06-28 20:00 / 2026-01-01 05:00 |
| Strictly increasing contracts | 80 / 80 |
| Duplicate timestamps | 0 |
| Off-hour timestamps | 0 |
| Non-one-hour steps | 0 |
| Missing or non-finite OHLCV cells | 0 |

The confusion comes from the legacy processed one-hour NPZ. It contains only
`train` and `test` tensors and does not preserve timestamps or row identities.
That legacy bundle must not be used for Phase 4. A new Phase 4 builder can and
should retain the raw `date` value for every eligible decision and target row.
Cross-frequency OHLC alignment indicates that this value labels the beginning
of the candle. Calendar eligibility must therefore use `date + 1h` as the
one-hour close availability time, rather than treating `date` itself as the
decision time.

## Capacity design

Two input lengths were audited:

- `seq_len=64`: 64 hours of input history;
- `seq_len=256`: 256 hours of input history, duration-matched to the existing
  four-hour `seq_len=64` inputs.

Targets use an eight-step horizon, equal to eight hours and therefore matched
to the earlier four-hour horizon of two steps. A task-training row is counted
only when its complete input window and target are before the walk cutoff. An
evaluation row is counted only when its decision is at or after the cutoff and
its target remains strictly inside that evaluation interval.

The completed Phase 3 four-hour encoder bundle had 21,696 training windows.
That count is used only as an empirical workload reference; overlapping windows
are correlated and are not equivalent to the same number of independent
observations.

## Duration-matched results

The following table reports the more conservative `seq_len=256` counts.

| walk policy | walks | walk | task-train rows | train contracts | evaluation rows | evaluation contracts | train rows / Phase 3 4h |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global calendar, rolling window | 2 | 1 | 118,708 | 59 | 28,592 | 20 | 5.47x |
| Global calendar, rolling window | 2 | 2 | 119,770 | 59 | 24,134 | 15 | 5.52x |
| Contract-relative, rolling window | 2 | 1 | 75,262 | 80 | 47,517 | 80 | 3.47x |
| Contract-relative, rolling window | 2 | 2 | 75,253 | 80 | 47,479 | 80 | 3.47x |
| Global calendar, rolling window | 3 | 1 | 118,708 | 59 | 26,388 | 20 | 5.47x |
| Global calendar, rolling window | 3 | 2 | 130,054 | 61 | 4,392 | 2 | 5.99x |
| Global calendar, rolling window | 3 | 3 | 97,511 | 55 | 21,930 | 15 | 4.49x |
| Contract-relative, rolling window | 3 | 1 | 75,262 | 80 | 31,462 | 80 | 3.47x |
| Contract-relative, rolling window | 3 | 2 | 75,241 | 80 | 31,475 | 80 | 3.47x |
| Contract-relative, rolling window | 3 | 3 | 75,254 | 80 | 31,419 | 80 | 3.47x |

At `seq_len=64`, the corresponding rolling training counts are still larger:
90,601--90,622 rows for three contract-relative walks and
108,073--141,777 rows for three global-calendar walks.

## Interpretation

The one-hour data is large enough in raw sample count to train the planned deep
encoders. Even with a duration-matched 256-step input and a fixed-width
training window, every audited training fold has at least 75,241 rows in the
relative analysis and 97,511 rows in the three-walk global-calendar analysis.
These are approximately 3.47--5.99 times the completed Phase 3 four-hour
training-row count.

This does not imply that all candidate walk schedules are equally sound. The
equally spaced three-walk global-calendar schedule has only two active
contracts in its middle evaluation interval. Its 4,394 highly overlapping rows
are not a representative pooled-market test set, and all of its audited
eight-hour targets are stable under the fixed `0.005` movement threshold. The
two-walk global-calendar schedule is healthier for this fixed cohort: its two
evaluation intervals contain 20 and 15 contracts and 28,592 and 24,134
duration-matched rows, respectively.

Therefore:

1. raw timestamps are available and permit genuine fixed-date splits;
2. one-hour data has adequate training capacity, including at `seq_len=256`;
3. two global-calendar walks are the better current starting point than three
   equally spaced walks for this cohort;
4. final cutoffs should be chosen under a predeclared coverage rule, not tuned
   for downstream model performance;
5. contract-relative walks remain useful lifecycle diagnostics, but they do
   not replace the global-calendar split for pooled deployment claims; and
6. the new Phase 4 artifact contract must preserve contract ID, input start,
   decision timestamp, target timestamp, and walk ID.

## Reproduction and artifacts

Run:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_1h_timestamp_capacity.py
```

Outputs are under
`experiments/phase4/data_analysis/top80_1h_timestamp_capacity/`:

- `contract_timestamp_audit.csv`
- `walk_capacity.csv`
- `training_capacity.png`
- `report.md`
- `summary.json`
- `artifact_hashes.json`

The script is diagnostic only and launches no training.
