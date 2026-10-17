# Phase 7C — Selected-variant branch analysis

**Date:** 2026-10-18

**Scope:** Owner-approved exploratory extension of Phase 7A. Implementation
supports configurable profiles under `src/analysis/`; the task roster is now
owner-approved and encoded. Target-runtime admission and real execution remain pending. SHAP is optional
and is not implemented by this extension.

## Question and selection boundary

How do standalone usefulness, practical removal contribution and geometric
similarity look in the framework configurations that performed best for each
task? This is post-hoc characterization of selected configurations, not an
independent confirmation of predictive superiority.

Choose exactly one existing framework per task, allowing the same framework
to be chosen for multiple tasks: one to three distinct variants in total.
Keep that task's configuration fixed across both calendar walks. Record the
previous comparison and metric/rule used for each choice before launching
analysis; do not automatically choose a winner or construct a metric-wise
envelope. The owner approved the following assignments on 2026-10-18:

| Task | Configuration | Change from H0 | Equal-walk epoch-50 selection metric |
|---|---|---|---|
| Classification | HC-ST | Replace contrastive CNN with Transformer (5 branches, 445 wide) | Highest mean macro-F1: 0.45518977 |
| Future price | HB-AL | Add BYOL LSTM (6 branches, 573 wide) | Lowest mean price MAE: 0.00746054 |
| Future realised variance | HC-AL | Add contrastive LSTM (6 branches, 573 wide) | Lowest mean MAE among noncanonical, non-duplicate variants: 0.0005661085 |

Selection reads the existing epoch-50 tables for
[Phase 6 temporal variants](../../experiments/phase6/encoder_variants/reports/complete_seed0/epoch50_by_walk.csv),
[Phase 6.5A capacity](../../experiments/phase6_5/lstm_capacity/reports/complete_seed0/epoch50_by_walk.csv),
and, for price only,
[Phase 6.5D residual CNN](../../experiments/phase6_5/residual_cnn/reports/complete_seed0/epoch50_by_walk.csv).
Each walk has equal weight; this is not pooled-row weighting or a best-per-walk
envelope. Duplicate-width controls are not selected scientific variants.
For variance, H0 has lower mean MAE (0.0005641149) than HC-AL. The owner
explicitly chose HC-AL to characterize the strongest noncanonical MAE variant;
do not describe it as better than H0 or as the RMSE winner.

Only analyse a configuration on the task for which it was selected; do not
cross all three configurations with all three tasks. These are existing
multi-branch framework variants, not task-specific end-to-end comparators.

## Required scope

- Reuse frozen encoders/feature stores and original seed-0 full-model heads;
  no encoder or full-reference retraining.
- Standalone and leave-one-out probes for every branch actually present in
  each selected framework, on its exact task rows in both walks.
- The same fixed 50-epoch probe recipe and 5/15/50 snapshots as Phase 7A;
  epoch 50 remains primary. Fresh-restart failed heads and retain verified
  completed runs under matching provenance.
- Standardized linear CKA between its branches, separately on training and
  evaluation samples for each task/walk. Keep each task population separate;
  cross-task CKA differences are not isolated architecture effects.
- Required metric/removal tables, dynamic-branch heatmaps, validation
  diagnostics, sampling manifests and a descriptive report.

For a task with B branches, two walks require `4 * B` probe trajectories.
The three-task inventory is `4 * (B_classification + B_price + B_variance)`:
60 if all selections have five branches, up to 72 if all have six. There
are six immutable full-framework references and twelve CKA matrices. Do not
deduplicate runs based on names alone or treat prior run files as reusable
without exact configuration/feature/row/scaler/recipe/code checks.

The approved 5/6/6-branch roster declares **68 probe trajectories** (20
classification, 24 price, 24 variance), 204 new probe snapshots and 18 existing
reference snapshots: **222 snapshots total**, plus twelve CKA matrices.

Conditional analyses, SHAP, extra seeds, pairwise combinations, uncertainty,
resource comparisons and further architectures are not completion gates.
The separate [Phase 7B plan](2026-10-18-phase-7b-full-model-shap-analysis-plan.md)
still owns full-H0 SHAP; optional selected-variant SHAP needs an explicitly
recorded explanation protocol before execution.

## Implementation and execution

### Owner-approved prelaunch numerical amendment (2026-10-18)

All 177 required inputs were SHA-256 verified on dev4. The first source audit
accepted five references but rejected HB-AL price Walk 1 at epoch 15: raw
predictions differed by at most `1.1921e-7`, while the auxiliary median Rank
IC changed from `0.13180367` to `0.13227349` (absolute `0.00046982`). Mean
Rank IC drift was `0.00001368`; primary MAE/RMSE drift was negligible and
within the original bounds. Near-tied ranks make this median discontinuous.

After reviewing that failure, the owner explicitly reaffirmed that small
numerical deviations should be recorded rather than block execution. This
is a disclosed revision after prelaunch replay, before any new Phase 7C head
training, not a silently predeclared threshold. The opt-in
`src/analysis/configs/phase7c_replay_policy.json` sets only
`rank_median_metric_atol=0.001`. Apply it with `--tolerances` and freeze it in
a new `experiments/phase7/selected_variants/dev4/` output root. All other
tolerances remain unchanged, including mean Rank IC, MAE/RMSE, raw outputs,
classification, Spearman and CKA. Raw-output errors, non-finite values,
structural/identity failures and hash/provenance mismatches remain invalid.
Accepted median drift is visibly logged with actual values and thresholds;
larger drift is still rejected.

Keep the original failed audit, policy and preparation under the parent
`experiments/phase7/selected_variants/` root intact. Do not rewrite source
predictions, metrics or checkpoints. This amendment does not change Phase
7A's archived 197/198 validation status or its frozen policy.

### Commands and persistence

Use the existing `analysis` package with `--profiles <json>` before the
subcommand. Its task-keyed profiles record `family`, `configuration` and
`selection_reason`. Families map to the existing canonical, Phase 6 temporal,
Phase 6.5A LSTM-capacity and Phase 6.5D residual-CNN registries; the last is
price-only. The approved file is
`src/analysis/configs/phase7c_selected_profiles.json`; `analysis --phase7c`
loads it explicitly. `--profiles <json>` remains available for custom files;
the two options are mutually exclusive. See `src/analysis/README.md` for
the schema and non-training audit/manifest commands.

The default without profiles remains canonical Phase 7A. Selected profiles
use separate `experiments/phase7/selected_variants/` outputs and frozen
profile manifests. Native feature/reference adapters retain canonical row
and target integrity, explicit legacy-path relocation, all source hashes,
original branch order, scaler parity and reference checkpoint replay. Missing
dependencies or material replay drift invalidate the affected entries rather
than being waived. Follow the established numerical-warning policy.

Execute sequentially on the owner-selected verified Lumid environment after
exact-commit synchronization, full dependency admission and focused smoke
checks. Both default `launch` and `run` remain non-training without explicit
`--execute`; no production run is authorized or claimed by this code update.

## Completion and reporting

The three assignments are recorded; preparation freezes them in the output
root and rejects subsequent changes. Complete
the dynamically declared standalone/removal matrix, full-reference replay,
CKA sampling/replay, plots and report. Explain each removal relative to its
own selected full framework, not automatically relative to H0. Preserve the
width/head-capacity confound and seed-0/post-hoc limitations; similarity is
not proof of redundant or unique predictive information.

This phase does not alter Phase 7A's archived 197/198 validated-snapshot
status, its outstanding numerical exception or its completion conditions.
