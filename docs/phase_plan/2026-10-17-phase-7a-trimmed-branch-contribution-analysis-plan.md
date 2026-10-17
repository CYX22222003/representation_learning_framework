# Phase 7A Trimmed Canonical Branch Contribution and Overlap Analysis Plan

**Date:** 2026-10-17  
**Status:** Current Phase 7A authority, adopted by the owner on 2026-10-17.  
**Execution status:** Implementation and execution have not started.  
**Grouping refinement:** The owner retained only calendar walks, contracts,
and future realised-variance strata; starting-probability and imputation groups
are deferred.
**Analysis refinement:** Prediction-error overlap and resource analysis are
excluded from the current study. Bootstrap uncertainty analysis is optional
future work, allowing time for the report and presentation. The required
analyses are ablations, CKA similarity, and the three agreed condition groupings.
**Supersedes:**
`2026-10-12-phase-7a-ablation-and-branch-complementarity-analysis-plan.md`
and `2026-09-26-phase-7a-representation-ablation-plan.md`.

## 1. Decision and purpose

Phase 7A is reduced to a defensible undergraduate-scale study of the five
canonical representation branches. The required scope is now Package A plus a
lean analysis package, B-Core. The ten pairwise neural probes and the heavier
structural analyses from the 2026-10-12 plan are deferred rather than required.

The central research question is:

> Which canonical branches are useful independently, and which make a
> consistent incremental contribution to the complete representation across
> downstream tasks and calendar walks?

The study retains complete task and walk coverage because cross-task and
cross-time consistency are central to the representation claim. Scope is cut
along configuration and diagnostic axes instead: no encoder is retrained, no
new external baseline is added, and no evaluation result may expand or select
the matrix.

Phase 6.8, Phase 6.6, and Phase 7B do not gate this study. The simple concat
probe remains the representation experiment. This plan does not authorize
pairwise-complementarity, causal-importance, architecture-selection, SOTA, or
profitable-alpha claims.

## 2. Required and deferred work

| Package | Contents | Status | New head trajectories |
|---|---|---|---:|
| A | Five single branches and five leave-one-out configurations on three tasks and two walks | Required | 60 |
| B-Core | Source/H0 replay, main metrics, removal effects, standardized CKA, and calendar-walk/contract/future-RV groups | Required | 0 |
| C | All ten two-branch concatenations | Deferred future extension | 60 if later commissioned |
| D | Duplicate-width pair controls | Deferred future extension | 120 if later commissioned |
| E | Additional head seeds | Deferred future extension | Not part of current scope |
| B-Extended | Effective rank, spectra, directional reconstruction, descriptor probes, named subblocks, raw-feature sensitivities, exhaustive conditions, bootstrap-length sensitivities, and inference microbenchmarks | Deferred future extension | 0; auxiliary fits only |

Required Phase 7A therefore contains **60 new seed-0 trajectories**, **180 new
snapshots**, and **six immutable H0 reference trajectories with 18 reference
snapshots**. Completing A without B-Core, or B-Core without all A runs, does
not complete the trimmed study.

The exact cross-product planning inventory is stored beside this plan in
`2026-10-17-phase-7a-trimmed-experiment-inventory.json`; expanding its ten
configurations, three tasks, and two walks produces exactly 60 run IDs.

## 3. Research questions and evidence

| ID | Question | Required evidence |
|---|---|---|
| Q1 | What can each branch predict by itself? | Single-branch task performance and skill against fixed non-learned references |
| Q2 | What changes when each branch is removed from H0? | Paired H0-versus-removal metric differences |
| Q3 | Which branches have similar representation geometry? | Standardized linear CKA |
| Q4 | Is contribution task-, calendar-walk-, contract-, or future-RV-tail-specific? | Complete task/walk tables, contract summaries, and future realised-variance strata |

Standalone usefulness, incremental contribution, and geometric similarity are
distinct. High CKA does not prove redundant task information. Pairwise
practical complementarity remains unresolved unless deferred Package C is
separately commissioned and completed.

## 4. Frozen branches, data, and tasks

The canonical branch order and widths remain:

| Symbol | Stored key | Meaning | Width | H0 coordinates |
|---|---|---|---:|---|
| S | `statistical` | AR/residual and GARCH summaries | 70 | `[0,70)` |
| T | `transformed` | FFT and Haar features | 55 | `[70,125)` |
| V | `vae` | VAE latent mean | 64 | `[125,189)` |
| C | `contrastive` | Contrastive CNN embedding | 128 | `[189,317)` |
| B | `byol` | BYOL CNN embedding | 128 | `[317,445)` |
| H0 | all five | Canonical concat | 445 | `[0,445)` |

Reuse the two accepted Phase 5 global-calendar walks, their separate encoders,
64-hour contexts, bounded isolated-one-hour fill, causal activity rule,
observed endpoints, target maturity, and longer-gap sequence breaks. Later
walk information never updates an earlier-walk model.

| Task | Target | Canonical source |
|---|---|---|
| `classification_h2` | DOWN/STABLE/UP at `h=2`, `tau=0.001` | Phase 5 walk data and canonical five-branch features |
| `absolute_price_h8` | `close[t+8h]` | Phase 5 h8 add-on data and features |
| `realised_variance` | Eight-hour strictly future realised variance from raw probability changes | Phase 6 H=8 labels and canonical features |

All configurations within one task/walk use identical ordered rows and
targets. Task populations remain different and are never treated as one common
prediction problem. The six source stores and six H0 runs must pass independent
identity, hash, recipe, prediction, and metric replay before any new run.

## 5. Required neural experiment matrix

### 5.1 Single branches

| ID | Included branches | Width |
|---|---|---:|
| `A-S` | S | 70 |
| `A-T` | T | 55 |
| `A-V` | V | 64 |
| `A-C` | C | 128 |
| `A-B` | B | 128 |

### 5.2 Leave-one-out configurations

| ID | Included branches | Width |
|---|---|---:|
| `A-noS` | T,V,C,B | 375 |
| `A-noT` | S,V,C,B | 390 |
| `A-noV` | S,T,C,B | 381 |
| `A-noC` | S,T,V,B | 317 |
| `A-noB` | S,T,V,C | 317 |

Each of the ten configurations runs on three tasks and two walks:
`10 x 3 x 2 = 60` trajectories. H0 is replayed, not retrained. No temporal
encoder branch, raw input, external representation, gated aggregator, pairwise
configuration, or result-selected subset enters the matrix.

### 5.3 Shared probe recipe

- Seed 0; Adam; learning rate `1e-4`; weight decay 0; batch size 512.
- One continuous 50-epoch run; retain epochs 5, 15, and 50; epoch 50 is primary.
- Deterministic training shuffle and retained remainder batch.
- Fit coordinate mean and population standard deviation on the complete
  supervised training population for that task/walk only.
- Replace scales below `1e-8` with one, standardize, and clip to `[-10,10]`.
  Save means, raw standard deviations, applied scales, clip limits, and hashes.
- Preserve constant coordinates in neural probes and report their counts.
- Use concat without trainable aggregation and the existing
  `d -> 128 -> 64 -> output` task-head family.
- Classification retains train-prior logit-adjusted cross-entropy and dropout
  0.2; price retains sigmoid output, endpoint MSE, and dropout 0.1; variance
  retains Softplus, Smooth L1 on `10000*RV`, and gradient clipping at 5.
- Preserve every other audited source-task optimizer, output, and metric
  semantic. There is no validation split, early stopping, or evaluation-driven
  checkpoint/configuration selection.

## 6. Required B-Core analysis

### 6.1 Main metrics and contribution definitions

Epoch 50 is the principal table; epochs 5 and 15 remain replayed supporting
results. Main-text primary metrics are:

- Classification: macro-F1 and balanced accuracy.
- Price: MAE, RMSE, and implied-movement mean cross-sectional Rank IC.
- Realised variance: MAE, RMSE, and Spearman correlation.

Retain the full existing supporting metric definitions in machine-readable
artifacts and appendices. Report non-learned references already frozen for each
task; do not add a new trained baseline.

For a higher-is-better metric `M` and lower-is-better error `E`, define the
benefit-positive removal contribution:

```text
Delta_j(M) = M(H0) - M(H0 without j)
Delta_j(E) = E(H0 without j) - E(H0)
```

Report absolute differences and relative error degradation when `E(H0)>0`.
A negative contribution means the removal improved the observed metric; it
does not prove the source information is intrinsically harmful. Different
input widths and first-layer parameter counts remain disclosed confounders, so
the result is practical incremental contribution rather than causal feature
importance.

### 6.2 Standardized linear CKA

CKA is the sole required structural diagnostic. Compute it only on the
classification population, separately for each walk and for training and
evaluation:

- Ten unique branch pairs.
- Two walks and two splits.
- **40 required standardized CKA scores**.

Use up to 4096 condition-balanced rows with PCG64 seed 170012 and the stable
round-robin sampling rule from the superseded 2026-10-12 plan. Freeze sampled
identities and hashes. Use the task-training-standardized, clipped coordinates;
remove only coordinates whose complete-training standard deviation is below
`1e-8` from this diagnostic. Compute centered linear CKA in float64.

Raw-feature CKA, price/RV population repeats, named AR/GARCH/FFT/Haar subblock
CKA, spectra, and effective rank are deferred.

### 6.3 Limited condition reporting

Use exactly these three reporting dimensions, considered separately rather
than crossed into a subgroup grid:

1. Walk 1 and Walk 2 separately before any pooling.
2. Natural row-weighted and equal-contract summaries.
3. For realised variance only, target rows below the training 90th percentile,
   from the 90th to below the 99th, and at or above the 99th percentile; also
   report tail row fraction, MAE, RMSE, bias, and SSE share at q90/q99.

Calendar means the existing Walk 1 versus Walk 2 comparison; no additional
monthly, weekly, or event-scenario performance tables are required. Future-RV
thresholds come only from that walk's task-training targets. Evaluation groups
are assigned retrospectively from observed future RV and never enter model
inputs, eligibility, or model selection.

Report counts and sparse-group warnings. Starting-probability, imputation,
lifecycle, activity, historical-RV, volume, and calendar-block subgroup tables
are deferred. Imputation metadata and the approved data-quality/eligibility
checks remain part of the underlying data contract. No weekly resampling or
additional calendar-block performance tables are required.

Pooled results may be produced only after walk-local results, by concatenating
already out-of-future predictions while retaining walk in each identity.
Recompute nonlinear metrics from pooled rows; do not average walk RMSE or F1.

## 7. Reporting contract

Use `experiments/phase7/ablation/` and retain the existing task/config/walk/seed
layout. Required report products are:

1. Complete epoch-50 task/walk/configuration tables, with 5/15 results in a
   supporting table.
2. Single-branch skill against fixed references.
3. H0 removal-contribution tables and plots with point estimates.
4. Four standardized CKA heatmaps and their sampled-identity manifests.
5. Calendar-walk, contract, and future-RV-stratum/tail tables with counts.
6. A four-question discussion corresponding to Q1--Q4 and separating
   measured evidence from hypotheses.

Every prediction retains ordered identities, targets, predictions, and logits
or regression-unit metadata. Undefined metrics use null plus a reason. No
invalid value is silently replaced with zero.

Report observed effect sizes and consistency across tasks, walks, and
contracts descriptively. Without bootstrap intervals or repeated seeds, do not
claim statistical significance, equivalence, or an established noise margin.
Two walk results are not independent training-seed replications.

## 8. Implementation and execution order

1. Record this trimmed scope and the exact 60-run inventory in a versioned
   implementation manifest. No deferred package is silently activated.
2. Replay the six task/walk source stores and six H0 references; record exact
   hashes, rows, recipes, and metrics.
3. Freeze branch maps, task/walk identities, seed/budget, main metrics, CKA
   samples, future-RV condition edges, and expected artifacts.
4. Build exact branch views and prove equality to source coordinates. Fit
   training-only scalers and test evaluation-perturbation invariance.
5. Run CPU integration tests for distinct width/task families, identity and
   branch-map failures, constants, invalid metrics, occupied paths, resume,
   CKA and the declared subgroup metric formulas.
6. Generate manifest-only launch artifacts. Require explicit `--execute`, a
   bounded runtime feasibility check, persistent logs, and resumable
   trajectories. This execution check does not require a resource-comparison
   experiment or resource report.
7. Execute all 60 runs without result-driven truncation.
8. Replay checkpoints, scalers, coordinates, predictions, metrics, identities,
   hashes, CKA samples, and subgroup assignments.
9. Generate every declared table and figure, including negative and unavailable
   cells.
10. Write the outcome discussion and update schedule/research documents only
    from executed evidence.

Planning, code existence, smoke tests, or a partial matrix do not establish
experimental completion.

## 9. Completion criteria

Phase 7A is complete only when:

- All 60 new trajectories and 180 snapshots replay.
- All six immutable H0 references and 18 reference snapshots independently replay.
- Every single/removal configuration covers all three tasks and both walks on
  exact task identities.
- All 40 standardized CKA values and their sample manifests replay.
- Required condition, q90/q99 tail, and pooled outputs
  exist with counts and availability reasons.
- The report discloses width/parameter confounding, seed-0 limitations,
  absence of statistical uncertainty estimates, retrospective cohort/pruning
  assumptions, negative results, and the fact
  that pairwise practical complementarity remains untested.

The final supported contribution is a task- and time-specific account of
standalone branch usefulness, practical incremental contribution, and geometric
similarity. It does not establish causal importance, pairwise
synergy, universal rankings, full-pipeline seed robustness, SOTA, or profitable
trading.

## 10. Deferred extension boundary

### 10.1 Optional uncertainty analysis if time permits

Bootstrap uncertainty analysis is not required for Phase 7A completion. If the
owner restores it, record its scope before calculating intervals and apply it
to the full declared comparison set rather than selecting favourable results.
Reuse saved predictions; this extension requires no new head training.

The retained recipe is seven-day UTC blocks anchored at each evaluation start,
2000 synchronous resamples with replacement using PCG64 seed 170013, and
pointwise 2.5th/97.5th percentile intervals for paired metric differences.
Keep all contracts in a sampled block together and retain duplicate-block
multiplicity. Use the same draws for all configurations within a task/walk,
recompute nonlinear metrics inside each draw, and preserve common valid
timestamp sets for Rank IC. Withhold intervals with fewer than five populated
blocks or fewer than 90% valid draws. This measures temporal evaluation-sample
variability conditional on fitted models, not head/encoder seed variability.
The superseded 2026-10-12 plan retains the detailed estimator specification.

### 10.2 Other deferred extensions

The following require a separately dated amendment and cannot be selected from
Phase 7A evaluation results:

- Any of the ten pairwise branch heads.
- Prediction-error overlap, residual-error correlations, and classification
  error-agreement/disagreement tables.
- Resource analysis: training-time, memory, storage, parameter-cost tables,
  and performance-cost comparisons. Feature widths and architecture shapes
  remain provenance needed to reproduce the ablations, not a resource study.
- Duplicate-width controls.
- Additional head or encoder seeds.
- Effective rank/spectra, directional reconstruction, descriptor probes, or
  named deterministic subblock diagnostics.
- Starting-probability and imputation subgroup reporting, expanded subgroup
  grids, alternative bootstrap block lengths, inference
  microbenchmarks, or Pareto analysis.
- Any promoted subset, new architecture, gated aggregator, alpha model, or
  trading strategy.

The 2026-10-12 document is retained as the fully specified deferred extension
design. Its A+C pairwise matrix and extended diagnostics are not completion
requirements for the current trimmed Phase 7A.

