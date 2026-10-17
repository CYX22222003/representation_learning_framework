# Phase 7A Analysis Experiments: Implementation Plan

**Date:** 2026-10-17  
**Status:** Implementation plan only; source implementation and experiment
execution have not started.  
**Scientific authority:**
[trimmed Phase 7A plan](2026-10-17-phase-7a-trimmed-branch-contribution-analysis-plan.md).  
**Planning inventory:**
`2026-10-17-phase-7a-trimmed-experiment-inventory.json`.

## 1. Deliverable and scope

Implement the four agreed analysis experiments:

1. Five single-branch probes for standalone usefulness.
2. Five leave-one-out probes for incremental contribution relative to H0.
3. Linear CKA between the five canonical branches.
4. Conditional performance by calendar walk, contract, and future realised
   variance; the last grouping applies only to the variance task.

All new utilities, data classes, configuration definitions, source adapters,
training lifecycle, analysis logic, replay, plotting, reporting, and command
orchestration belong under **`src/analysis/`**. Existing scientific primitives
may be imported from their current modules. Tests belong under
`tests/analysis/`, plans under `docs/phase_plan/`, and generated artifacts under
`experiments/phase7/ablation/`.

Use a package CLI, `PYTHONPATH=src .venv/bin/python3 -m analysis`, so no new
experiment logic or launcher generation is needed under `scripts_v*/`.
The commands described below are proposed interfaces, not existing commands.

Do not implement bootstrap uncertainty, prediction-error overlap, resource
comparisons, probability/imputation groups, pairwise probes, or new encoder
training as part of this work. Bootstrap uncertainty remains optional future
work. The current report contains descriptive effects without significance or
equivalence claims.

## 2. Repository findings that affect the implementation

The repository already supplies:

- Canonical named feature arrays, branch dimensions and identity fields in
  `src/features/phase5_features.py`.
- Original H0 validators in `src/training/phase5_downstream.py`,
  `src/training/phase5_absolute_price.py`, and
  `src/training/phase6_volatility.py`.
- Width-parameterized `TrendClassifier`, `AbsolutePriceRegressor`, and
  `VolatilityRegressor` heads.
- Logit-adjusted loss and probabilistic classification metrics under
  `src/tasks/phase2_classification/`.
- Regression/RV metric formulas in the original task trainers.
- `centered_linear_cka` in `src/evaluation/phase6_encoder_variants.py`.
- Resume-state handling examples in the Phase 6.9 training lifecycles.

Important adaptation boundaries:

- Phase 5's standardizer requires width 445. The Phase 6 variant standardizer
  accepts only 445/573 and its configuration registry is temporal-study-specific.
  They cannot directly load the eight distinct widths in this study.
- The Phase 6 variant runner rejects any occupied run directory and lacks a
  full resumable lifecycle. Build the Phase 7 lifecycle under `src/analysis/`.
- Existing price breakdown helpers automatically generate probability,
  imputation, and lifecycle groups that were removed from this scope. Reuse
  metric formulas, not those full reporting wrappers.
- The old CKA sampler selects the first encoder-training rows. Phase 7 instead
  requires a fixed condition-balanced sample from supervised classification
  features, including a separate evaluation sample.
- The old Rank IC helper counts rows per timestamp. Phase 7 must enforce five
  distinct conditions, reject duplicate condition/timestamp rows, and align
  valid timestamps in comparisons.

Leave completed experiment source and artifacts intact. New adapters must not
relax validators or mutate original H0 runs to make them fit this study.

## 3. Proposed package and responsibilities

```text
src/analysis/
  __init__.py
  __main__.py          # package entry point
  cli.py               # commands and arguments; delegates all work
  types.py             # all analysis data classes
  contracts.py         # branches, matrix expansion, fixed task recipes
  sources.py           # original datasets/features/H0/scalers and adapters
  features.py          # branch views and training-only standardization
  tasks.py             # head/loss factories, prediction and metric adapters
  training.py          # train/resume/infer lifecycle
  ablation.py          # single-branch skill and leave-one-out effects
  similarity.py        # balanced sample, CKA, sample/score replay
  conditional.py       # walk/contract/RV grouping and grouped metrics
  artifacts.py         # paths, hashes, atomic I/O, manifests, run state
  replay.py            # independent checkpoint/prediction/metric validation
  reporting.py         # complete tables, figures, report assembly
```

Keep this one small package rather than building a generic experiment platform.
`types.py` and `contracts.py` define shared interfaces; analysis modules consume
validated feature or prediction records, not raw ad hoc dictionaries.
`reporting.py` consumes validated metric rows and analysis artifacts, and never
trains or selects models. `cli.py` exposes orchestration without scientific
logic. Existing task modules do not import the new analysis package.

## 4. Data classes and validation boundaries

Use frozen data classes for metadata; mark stored NumPy arrays read-only or
defensively copy them, since a frozen data class alone does not freeze arrays.

| Data class | Main fields | Required checks |
|---|---|---|
| `BranchSpec` | key, symbol, width, H0 coordinate interval | Canonical order/width; no duplicate keys |
| `AblationSpec` | configuration ID, kind, retained branches, removed branch | Exactly one branch for singles or exactly four for removals; calculated width |
| `TaskWalkSource` | task, walk, data/feature/manifest paths, H0 run/scaler paths | Expected source family, files, hashes, task and walk |
| `RowIdentity` | walk, condition IDs, window starts, decision dates, availability, task target identity | Equal lengths, ordered uniqueness, exact source equality |
| `TaskSplit` | split name, identities, targets, named feature arrays, required task metadata | Finite arrays, matching rows, correct labels/units; no target-based filtering |
| `FeatureStandardizer` | mean, raw std, applied scale, clip limits, training identity hash | Training-only population; declared width and finite state |
| `ProbeRunSpec` | task, walk, ablation, seed, budget, batch/optimizer/loss recipe, destination | Frozen recipe; H0 cannot enter training mode |
| `PredictionRecord` | run/snapshot identity, row identity, targets, logits or predictions, units, hashes | Exact row/target pairing; output shape/range; verified snapshot |
| `ConditionSpec` | walk/contract rules, RV q90/q99, training-target hash | No removed group types; RV thresholds are walk-training-derived |
| `MetricResult` | metric name, value, direction, counts, availability/reason | Explicit null/reason for undefined metrics; no NaN JSON |

Use existing `IDENTITY_FIELDS` for source-row matching and augment them with
walk/task and each source task's target identity or maturity metadata. A
matching row count alone is insufficient. Do not join tasks to a smaller
intersection: each task retains every original eligible row.

## 5. Source inventory and H0 reuse

Paths are relative to the repository root; `{w}` is 1 or 2.

| Task | Data bundle | Canonical feature store |
|---|---|---|
| `classification_h2` | `experiments/phase5/data_preparation/walk{w}/market_1h_seq64_h2.npz` | `experiments/phase5/features/walk{w}/five_branch_epoch50.npz` |
| `absolute_price_h8` | `experiments/phase5/downstream_addons/shared/h8/data/walk{w}/market_1h_seq64_h8.npz` | `experiments/phase5/downstream_addons/shared/h8/features/walk{w}/five_branch_epoch50.npz` |
| `realised_variance` | `experiments/phase6/volatility_prediction/data_preparation/walk{w}/volatility_1h_seq64_h8.npz` | `experiments/phase6/volatility_prediction/features/walk{w}/five_branch_epoch50_h8.npz` |

H0 locations:

- Classification runs:
  `experiments/phase5/downstream/walk{w}/classification/seed0/`;
  scaler: `experiments/phase5/downstream/walk{w}/feature_standardizer.npz`.
- Price runs:
  `experiments/phase5/downstream_addons/tasks/absolute_price_h8/walk{w}/seed0/`;
  scaler: `experiments/phase5/downstream_addons/shared/h8/feature_scalers/walk{w}/feature_standardizer.npz`.
- RV runs:
  `experiments/phase6/volatility_prediction/runs/framework_h0/walk{w}/seed0/`;
  scaler: the original run's `feature_standardizer.npz`.

`sources.py` validates all six stores with the applicable original validators,
audits their dependencies, and independently replays all six H0 trajectories
at 5/15/50. Record source/scaler/checkpoint/target hashes, actual row counts,
recipe and identity validation under the new analysis artifact root.

Normalize existing H0 snapshot schemas into `PredictionRecord` in memory or
new derived records. Preserve the source files. Recompute required metrics
from predictions when an original summary lacks them, such as common-timestamp
price Rank IC. Existing stored metrics remain their historical records.

## 6. Feature preparation and fixed training

### 6.1 Manifest and feature views

Expand the existing JSON planning inventory into 60 new entries and six H0
reference entries. Each new entry fixes data/feature source hashes, identities,
branch order/coordinate map, target units, recipe, and output path.

Keep six shared source handles/caches, loaded one task/walk at a time. Select
named branch arrays without recomputing encoders or copying full stores for
each configuration. Persist compact view manifests and hashes; materialize only
the configuration currently needed. Preserve canonical coordinate order.

Support the eight distinct new input widths: 55, 64, 70, 128, 317, 375, 381,
and 390. Prove selected coordinates equal the source arrays exactly.

Fit each task/walk configuration scaler using its full training features in
float64, population std, threshold `1e-8`, unit fallback scale, and clipping
`[-10,10]`; materialize head input as float32. Verify the values match the
corresponding coordinates of the canonical task/walk scaler. Audit any mismatch
before training; do not silently rescale a source. Save array hashes rather
than expecting regenerated NPZ container files to have identical byte hashes.

### 6.2 Head and loss adapters

Reuse the established head classes with the selected `input_dim`:

| Task | Output and objective | Additional recipe |
|---|---|---|
| Classification | 3 logits; `CE(logits + log(training_priors), label)` | GELU, dropout 0.2; infer with unadjusted logits |
| Price | Scalar sigmoid; MSE against raw `close[t+8h]` | GELU, dropout 0.1; no current-price skip |
| RV | Scalar Softplus; Smooth L1 beta 1 against `10000*RV` | GELU, no dropout; gradient norm clip 5; divide output by 10000 for reporting |

All heads have hidden widths 128 and 64. Seed 0, Adam `lr=1e-4`, weight decay
0, batch 512, deterministic training shuffle with a saved generator, retained
remainder, 50 epochs, and snapshots 5/15/50. No additional clipping for price
or classification. No validation, early stopping, or evaluation-based choice.

### 6.3 Resumable lifecycle

Represent state as planned, initialized, training, trained, evaluated, and
replay-valid; a directory or completion marker alone is not proof of validity.

Save atomic epoch-boundary resume state containing model, optimizer, completed
epoch, training loss history, Python/NumPy/Torch CPU/CUDA RNG states, shuffle
generator state, recipe/source/scaler hashes, and relevant code fingerprints.
On interruption, restart from the last committed epoch. Reconcile missing
5/15/50 snapshots from the same committed state without rerunning a completed
epoch or consuming extra training RNG state.

Initialize resume state before the first epoch; enforce a single writer per run.
Refuse conflicting occupied paths. Skip a completed run only after replay;
resume an incomplete run only after exact provenance checks. Corrupt or mismatched
state is a failure, not an instruction to overwrite or start a new run silently.

Save training snapshots first; run evaluation after training so inference or
head reconstruction cannot affect subsequent dropout/RNG progression. Recover
missing prediction/metric outputs from existing valid snapshots. Completing
training and completing replay are separate states.

## 7. Four analysis implementations

### 7.1 Singles and leave-one-out

`ablation.py` produces all task/walk/snapshot metric rows from verified
`PredictionRecord` objects. H0 and candidates must match in ordered identities
and targets before any comparison. Main report is epoch 50; 5/15 form a
supporting table. Do not select a best epoch or branch subset.

- Singles: absolute metrics, descriptive H0 difference, and skill against the
  task's fixed non-learned references.
- Removals: benefit-positive contribution is `M(H0)-M(no_branch)` for a
  higher-is-better metric, or `E(no_branch)-E(H0)` for an error. Include relative
  error degradation when the H0 error is positive.
- Classification: macro-F1 and balanced accuracy primary, fixed class order
  DOWN/STABLE/UP; retain supporting source metrics.
- Price: MAE, RMSE, implied-movement mean cross-sectional Rank IC primary.
- RV: MAE, RMSE, Spearman primary, in raw squared-probability units.

Preserve fixed references: STABLE/training priors for classification;
current-price persistence and last-hour reversal ranking for price; zero,
training RV median, and historical eight-hour RV for variance. Recompute each
reference on exactly the candidate's rows. Price persistence has undefined
movement Rank IC; reversal is the fixed ranking comparator rather than a
calibrated future-price forecast.

The analysis-owned Rank IC adapter requires at least five distinct conditions,
average ranks for ties, and nonconstant scores/targets. Save per-timestamp IC
and coverage. For H0-versus-removal and single-versus-H0 comparisons, use the
common valid timestamps for that declared comparison and show individual
coverage too. Per-contract cross-sectional IC is unavailable, not replaced by
a single-contract time-series correlation.

### 7.2 CKA similarity

`similarity.py` uses only the classification feature population, in the fixed
order walk1/train, walk1/test, walk2/train, walk2/test.

Sort conditions and rows by stable identity, permute within each condition
using a single PCG64 seed 170012 stream in that processing order, then select
rows in sorted-condition round-robin order until 4096 rows or exhaustion.
Persist full identities, sample positions/counts, algorithm version, and hashes.
Use the same sample for all five branches in a task/walk/split.

Apply the classification training scaler to both splits. Remove only coordinates
whose full-training std is below `1e-8` for CKA; retain them in neural probes.
Call the existing float64 centered linear CKA primitive through an adapter.
Convert collapsed/undefined inputs to explicit unavailable results with reasons.
Never silently use diagonal one for a collapsed branch.

Produce four 5x5 matrices: **40 unique off-diagonal pair scores**, with four
heatmaps. No new encoder inference, raw-geometry sensitivity, extra task
population, or prediction-error similarity experiment is needed.

### 7.3 Conditional performance

`conditional.py` uses epoch-50 predictions and the same task metric adapter.
Groups are separate dimensions, not a cross-product grid.

- Calendar: Walk 1 and Walk 2. Retain each walk's separately fitted state.
- Contract: all condition IDs. Report natural row-weighted metrics and the
  equal-weight mean of per-contract metrics, excluding undefined values with
  reasons/counts. Contract-macro RMSE is the mean of contract RMSEs.
- Future RV: RV task only. Freeze q90/q99 with NumPy linear quantiles on that
  walk's complete training targets; assign test groups as `<q90`,
  `q90<=y<q99`, and `y>=q99`. Collapsed thresholds produce explicit empty bins.
  Also report cumulative q90/q99 tail MAE, RMSE, bias and SSE share.

RV groups are retrospective outcome diagnostics. They never enter inputs,
eligibility, scaling, or training. Persist thresholds, training-target hashes,
group counts, and row-mask hashes. Keep all observations, including extreme ones.

Show sample/contract counts and missing-class or undefined-correlation flags.
Retain fixed-three-class classification scores and the all-three-classes
contract subset as a supporting summary. Flag groups below 100 rows or three
contracts as sparse; per-contract results are explicitly case-specific.
Do not add a five-calendar-block requirement after calendar-block reporting
was removed.

After separate walk results, recompute pooled metrics from concatenated
out-of-future predictions with walk-aware identities. Also label equal-walk
averages separately. Pooled contract groups use `(walk, condition)`. Reject
duplicate evaluation identities or overlapping evaluation periods. Never
substitute an average of walk F1/RMSE for recomputed pooled metrics.

### 7.4 Report generation

`reporting.py` generates:

1. Complete task/walk/configuration/snapshot tables and an epoch-50 main table.
2. Single-branch reference-skill and H0-removal-effect tables.
3. Removal-effect plots with labelled metric direction; no confidence bars.
4. Four CKA heatmaps.
5. Calendar/contract/RV-tail tables and compact comparison plots.
6. A Markdown report organized around the four research questions, with main
   findings separated from speculative explanations and limitations.

Export numeric JSON/CSV plus standalone PNG/SVG figures for the report and
presentation. Keep required numeric outputs complete; avoid a figure for every
contract/configuration. Strict JSON stores null plus a reason for undefined
values. No resource or error-overlap table is generated.

## 8. Artifact layout

```text
experiments/phase7/ablation/
  manifests/
    source_audit.json
    matrix_seed0.json
    reference_replay.json
    condition_spec.json
  features/<task>/walk<w>/
    branch_views.json
  downstream/<task>/<config>/walk<w>/seed0/
    config.json
    dataset_manifest.json
    architecture_manifest.json
    feature_standardizer.npz
    resume.pth
    history.npz
    training_complete.json
    e{5,15,50}/checkpoint.pth
    e{5,15,50}/predictions.npz
    e{5,15,50}/metrics.json
    replay_validation.json
  analysis/
    structural_samples/classification_h2/walk<w>/<split>/
    cka/classification_h2/walk<w>/
    conditional/<task>/walk<w>/
  reports/canonical_five_branch_seed0/
    summary.md
    tables/
    figures/
```

Source canonical features and H0 runs remain external dependencies referenced
by hashes. All actual numerical inventory counts come from replayed sources,
not copied planning counts. Operational logs and model shapes are provenance;
they do not reinstate a resource-comparison study.

## 9. Proposed CLI and execution order

All commands run from the repository root with the existing WSL `.venv`.
Commands are to be implemented; none is executed by adopting this plan.

```bash
PYTHONPATH=src .venv/bin/python3 -m analysis audit
PYTHONPATH=src .venv/bin/python3 -m analysis prepare
PYTHONPATH=src .venv/bin/python3 -m analysis launch --device cpu
PYTHONPATH=src .venv/bin/python3 -m analysis validate --stage preparation
PYTHONPATH=src .venv/bin/python3 -m analysis similarity
PYTHONPATH=src .venv/bin/python3 -m analysis launch --device cuda --execute
PYTHONPATH=src .venv/bin/python3 -m analysis validate --stage runs
PYTHONPATH=src .venv/bin/python3 -m analysis conditional
PYTHONPATH=src .venv/bin/python3 -m analysis report
PYTHONPATH=src .venv/bin/python3 -m analysis validate --stage all
```

`audit` is read-only for original sources and records results under Phase 7.
`prepare` builds view/scaler/group manifests without training. `launch` defaults
to manifest generation and bounded synthetic CPU checks. Only `--execute`
trains/resumes real heads, after preparation validation and a bounded runtime
feasibility check. Similarity/conditional/report commands only process existing
features or predictions and must never trigger hidden training.

Support one configuration/task/walk for debugging or staged execution, but keep
the full frozen matrix and an incomplete-inventory warning visible. A final
report requires all 66 entries and 198 snapshots to pass replay. Partial reports
must be explicitly labelled partial. Restarting commands must be idempotent
when dependencies and settings match; mismatches fail with the exact cause.

## 10. Implementation sequence and acceptance checks

| Step | Implement | Acceptance condition |
|---|---|---|
| 1 | Types, contracts, source adapters, atomic artifact I/O | Exactly 60 new entries plus six references; all six sources/H0 references audited; rejected hash/identity/task/walk drift |
| 2 | Named views, scalers, task heads/losses | All eight widths work for all three tasks; source-coordinate/scaler parity; evaluation perturbation cannot change fitted state |
| 3 | Train/resume/predict/replay | Uninterrupted versus interrupted/resumed synthetic runs agree on the same backend; missing snapshot recovery; wrong provenance/occupied path rejected |
| 4 | Ablation metrics and comparisons | Known-answer errors/F1/skill/removal signs; exact target/identity pairing; correct RV unit inversion and Rank IC coverage |
| 5 | CKA sampling and matrices | Deterministic balanced identities; symmetry and self-similarity; collapse handling; replayable 40-pair inventory |
| 6 | Conditional metrics | Correct q90/q99 boundaries/ties, training-only thresholds, sparse/empty/missing-class behaviour, macro versus pooled formulas |
| 7 | CLI and report | Default commands train nothing; staged execution cannot be called complete; every output links to validated source predictions |
| 8 | Real matrix and handoff, in a subsequent execution task | 60 trajectories/180 new snapshots and six references/18 snapshots replay; 40 CKA scores and required group/report artifacts exist |

Keep tests under `tests/analysis/` focused on scientific validity and recovery:
use small deterministic arrays and known-answer metrics, plus bounded task-head
integration. Test report completeness and excluded-output absence rather than
pixel-perfect plots or generic framework scaffolding. Before real execution,
verify real-source loading/scaling/reference replay for each task/walk; this is
an integration gate, not evidence that new models were trained.

Stop adding analyses once the declared outputs are covered. The remaining work
then becomes interpretation, writing, and presentation. Implementation status
and execution status must be updated separately from repository evidence.
