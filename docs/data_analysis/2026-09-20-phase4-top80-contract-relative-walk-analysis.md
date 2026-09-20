# Phase 4 Top-80 Contract-Relative Walk Analysis

Date: 2026-09-20
Status: Retrospective lifecycle and sample-feasibility analysis; no model training

## 1. Corrected question

This analysis compares two walk-forward policies whose boundaries are defined
separately as fractions of each contract's own observed length:

1. **contract-relative anchored:** the training start remains at relative
   position zero while the endpoint expands; and
2. **contract-relative fixed-length:** both ends of a fixed-width relative
   training window advance by one evaluation fraction.

For an initial training fraction of `0.5` and three walks, the allocations are:

| walk | anchored train | fixed-length train | shared evaluation |
|---:|---|---|---|
| 1 | `[0, 0.50)` | `[0, 0.50)` | `[0.50, 0.667)` |
| 2 | `[0, 0.667)` | `[0.167, 0.667)` | `[0.667, 0.833)` |
| 3 | `[0, 0.833)` | `[0.333, 0.833)` | `[0.833, 1.00)` |

Relative position is raw row index divided by the contract's final raw index.
The training window and evaluation length therefore scale with each contract
rather than using one absolute calendar duration.

## 2. Validity boundary

This is a sound way to study lifecycle behaviour and whether a representation
trained on earlier relative periods transfers to later relative periods. It is
not, by itself, a leakage-safe simulation of deploying one pooled encoder:

- the same relative cutoff occurs on different calendar dates across contracts,
  so pooled training can include a later calendar observation from one contract
  while evaluating an earlier calendar observation from another; and
- the final observed contract length is future information unless the relevant
  termination boundary was known at the prediction time.

Consequently, all results here are labelled **retrospective lifecycle
evidence**. A final Phase 4 pooled-model experiment still needs global calendar
cutoffs, or another proof that every training observation was available before
every prediction it is used to evaluate. If a scheduled resolution timestamp
was known contemporaneously, time-to-resolution can later be used as a causal
feature; observed file length cannot silently substitute for that metadata.

## 3. Data and target contract

The analysis uses the Phase 3 prefix-ranked top 80 four-hour contracts. Source
hashes and raw row counts were replayed before analysis. The target definitions
are:

- signed probability movement: `close[t+2] - close[t]`;
- arithmetic return: `(close[t+2] - close[t]) / close[t]`; and
- labels: `DOWN < -0.005`, `STABLE` within `[-0.005, 0.005]`, and
  `UP > 0.005`.

Inputs contain 64 four-hour bars and the target horizon is two bars, or eight
hours. Every training input must lie completely inside its relative training
window. Every supervised target must fall strictly before the relative cutoff.
All evaluation identities and targets are hash-verified as identical between
the anchored and fixed-length analyses.

The top-80 selection remains retrospective for this diagnostic. It is not yet
a Phase 4 cutoff-local universe rule.

## 4. Three-walk results

Unlike the discarded absolute-calendar analysis, the corrected relative
scheme gives every contract representation in every evaluation fold:

| walk | evaluation fraction | rows | contracts | zero movement | STABLE | zero-baseline MAE | near boundary |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `[0.50, 0.667)` | 8,041 | 80 | 43.64% | 66.92% | 0.007300 | 42.47% |
| 2 | `[0.667, 0.833)` | 8,044 | 80 | 55.83% | 75.96% | 0.004781 | 53.62% |
| 3 | `[0.833, 1.00)` | 7,901 | 80 | 63.21% | 85.09% | 0.003796 | 67.17% |

The monotonic increase in persistence and boundary concentration independently
reproduces the earlier lifecycle finding on all 80 contracts.

### Movement labels

| walk | DOWN | STABLE | UP |
|---:|---:|---:|---:|
| 1 | 16.50% | 66.92% | 16.58% |
| 2 | 12.15% | 75.96% | 11.90% |
| 3 | 7.33% | 85.09% | 7.58% |

The fixed `tau=0.005` classification task becomes progressively more
imbalanced toward `STABLE`. Evaluation rows must retain their natural class
frequency. Any sampling or loss adjustment belongs exclusively to the
corresponding training window.

### Training availability

| walk | anchored rows | fixed-length rows | fixed / anchored | contracts in both |
|---:|---:|---:|---:|---:|
| 1 | 18,900 | 18,900 | 100.0% | 80 |
| 2 | 26,941 | 18,890 | 70.1% | 80 |
| 3 | 34,985 | 18,893 | 54.0% | 80 |

The fixed-length policy maintains approximately 18,900 supervised rows in
every walk. Every contract contributes at least 64 rows, although only 22 of
80 contracts contribute at least 256 rows to each fixed-length window. Because
the encoders are pooled across contracts, the aggregate sample count is large;
per-contract expert models would not have equally strong support.

## 5. Two-walk sensitivity

The two-walk allocation uses evaluation fractions `[0.50, 0.75)` and
`[0.75, 1.00)`. It provides 12,057 and 11,929 evaluation rows respectively,
with all 80 contracts in both intervals. Its fixed-length training windows
again contain approximately 18,900 rows.

The earlier argument for reducing from three walks to two was caused by the
incorrect absolute-calendar interpretation. Under contract-relative
boundaries, three walks no longer suffer from a sparse two-contract middle
fold. Both two and three walks are feasible by aggregate sample count:

- three walks provide finer lifecycle resolution and approximately 8,000
  evaluation rows per walk;
- two walks provide approximately 12,000 evaluation rows per walk and fewer
  encoder retraining runs; and
- neither option should be selected from downstream performance after the
  fact.

The three-walk schedule is therefore a viable descriptive default, while two
walks remain a compute-saving sensitivity rather than a remedy for sample
failure.

## 6. Probability movement versus arithmetic return

No selected row has an exactly zero current close, so arithmetic return is
defined throughout this cohort. It remains highly dependent on the starting
probability. Across the three evaluation intervals, the 99th percentile of
absolute return is approximately `0.83–1.00`, while probability movement stays
bounded and directly interpretable in probability points. Within the
`(0, 0.01]` starting-price band, a probability-movement MAE below `0.0005`
coexists with an absolute-return 99th percentile of `1.0`.

Signed probability movement remains the recommended primary regression target.
Arithmetic return is a secondary sensitivity task requiring price-band
reporting and a predeclared zero-price rule.

## 7. Current judgement

The corrected analysis supports contract-relative walks for these purposes:

- descriptive lifecycle analysis;
- checking whether fixed early-life representations transfer to later life;
- comparing an anchored-history representation with a recent-relative-window
  representation; and
- measuring lifecycle-specific target and class difficulty.

It does not support replacing global calendar evaluation for a pooled model's
final deployment claim. A defensible Phase 4 design can use both axes:

1. global calendar cutoffs establish what information was actually available;
2. relative lifecycle position is reported inside each global evaluation
   interval; and
3. this contract-relative analysis is presented separately as retrospective
   evidence about lifecycle non-stationarity.

## 8. Reproducibility

Three-walk analysis:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py --scheme relative_anchored
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py --scheme relative_window
.venv/bin/python3 scripts_v2/report_phase4_walk_data.py
```

Two-walk sensitivity:

```bash
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py \
  --scheme relative_anchored --walk-count 2 \
  --output-root experiments/phase4/data_analysis/top80_contract_relative_walk_forward/candidate_2walk
.venv/bin/python3 scripts_v2/analyze_phase4_walk_data.py \
  --scheme relative_window --walk-count 2 \
  --output-root experiments/phase4/data_analysis/top80_contract_relative_walk_forward/candidate_2walk
.venv/bin/python3 scripts_v2/report_phase4_walk_data.py \
  --root experiments/phase4/data_analysis/top80_contract_relative_walk_forward/candidate_2walk
```

Artifacts are stored under:

```text
experiments/phase4/data_analysis/top80_contract_relative_walk_forward/
```

Each scheme records its configuration, source and artifact hashes, selected
contracts, per-walk and per-contract counts, movement and return summaries,
class and lifecycle distributions, price-band return diagnostics, and visual
overview. No preprocessing bundle, encoder checkpoint, downstream model, or
baseline was trained.
