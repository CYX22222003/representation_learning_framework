# Phase 7B — Full-model SHAP analysis

**Date:** 2026-10-18

**Status:** Owner-approved scope; planning only. Implementation and execution
are intended for the next working session ("tomorrow"), not this documentation
update. No SHAP values have been computed.

## 1. Purpose and phase boundary

Add a small post-hoc explanation extension to Phase 7A: inspect which inputs
influence the existing full five-branch H0 models' predictions on the three
supervised downstream tasks. Do not add another training matrix.

Phase 7B now means this SHAP extension. The previous provisional "Phase 7B
alpha research" label is superseded; alpha research remains deferred and
unassigned, with no search or trading experiment authorized.

Phase 7A remains governed by its
[trimmed plan](2026-10-17-phase-7a-trimmed-branch-contribution-analysis-plan.md).
Its archived report at
`experiments/phase7/ablation/restarts/dev2/reports/canonical_five_branch_seed0/summary.md`
is PARTIAL / INCOMPLETE, with 197/198 validated snapshots. The excluded
price A-B/Walk 1/epoch-50 snapshot is not a full H0 model. Preserve that
exception; Phase 7B neither waives it nor changes Phase 7A completion gates.

## 2. Fixed model and input scope

Reuse the original immutable seed-0, epoch-50 H0 references used by Phase 7A,
not any single-branch or leave-one-out head. There are **six existing heads**:

| Task | Explained output | Walks |
|---|---|---|
| `classification_h2` | DOWN, STABLE and UP probabilities, each explained separately | 1 and 2 |
| `absolute_price_h8` | Predicted `close[t+8h]` in raw probability units | 1 and 2 |
| `realised_variance` | Predicted strictly future eight-hour RV in raw RV units, not `10000*RV` | 1 and 2 |

Explain all **445 representation coordinates** entering each existing head:
statistical 70, transformed 55, VAE 64, contrastive 128 and BYOL 128. Reuse
each head's original training-fitted scaling, clipping and inference/output
conventions. Keep task and walk populations separate.

The primary analysis is coordinate-level SHAP on the full model, not a
five-player grouped-Shapley experiment. Use deterministic feature names and
the original coordinate map. Neural latent coordinates are labels such as
`byol_017`, not identified economic factors. Branch-level aggregation is not
a required deliverable and must not be presented as exact grouped Shapley.

No encoder/head retraining, extra seeds, model selection, ablation SHAP,
raw-timestep attribution, architecture changes or external baseline runs.

## 3. Small, reproducible explanation protocol

Before computing production attributions, record an implementation manifest
that fixes:

- The explainer/estimator, dependency versions and masking/reference
  semantics appropriate for the existing neural heads. These choices are
  not yet selected by this scope-only plan.
- A bounded background sample from that task/walk's training rows only,
  plus a bounded evaluation sample from its existing held-out rows. Save
  deterministic sampling rules, seeds, row identities, counts and hashes;
  use the same explained rows for all three classification outputs.
- Background/evaluation sample sizes, estimator evaluation budget,
  batching/device and any approximation settings. Run tasks/walks
  sequentially; concurrency is unnecessary.
- Output units, expected/base values, and predeclared numerical tolerances
  for prediction replay and the applicable SHAP additivity/approximation
  checks. Record diagnostics rather than relaxing tolerances after results.

Choose settings for bounded runtime and correctness, not attractive feature
rankings. Do not claim exact Shapley values when using an approximation.
The background and masking choices affect interpretation, especially for
related branch representations; do not assume feature independence is true.

## 4. Required outputs

For every task/walk and every explained output, retain:

1. SHAP values, base values, original predictions, feature names and sampled
   row identities, with the explanation manifest and source hashes.
2. A global mean-absolute-SHAP importance bar plot and a SHAP beeswarm plot.
   Explain all 445 coordinates even if a fixed top-k is displayed; freeze
   the display policy before production calculation.
3. Class-labelled plots for each classification output, not one ambiguous
   mixture of DOWN/STABLE/UP attributions. Regression figures retain their
   native output units; attribution magnitudes are not comparable across
   different tasks' units.
4. Persistent sequential-stage logs, validation diagnostics, and a short
   descriptive report. Preserve failed/invalid outputs without treating
   them as valid explanations.

Reuse utilities under `src/analysis/`; any new explanation utilities, CLI,
plotting or reporting also belong there, with tests under `tests/analysis/`.
Keep Phase 7B outputs separate under `experiments/phase7/shap/`; do not
overwrite Phase 7A artifacts. Locate the original H0 checkpoints explicitly:
model weights are not included in the Git result archive. Record the chosen
verified runtime and exact source commit before execution. If Lumid is used,
follow the sandbox guide and persistent-storage rules.

## 5. Interpretation and completion

SHAP explains the trained model's predictions relative to the chosen
reference. Positive/negative attribution describes the direction of an
output shift, not whether prediction accuracy improves. Mean absolute SHAP
describes attribution magnitude, not a percentage of predictive performance.
It does not prove unique information, causal market mechanisms or branch
complementarity, and does not replace Phase 7A's ablations or CKA.

Complete Phase 7B only after all six H0 heads and their class-specific or
regression outputs have saved explanations, required plots, provenance,
appropriate numerical checks and a descriptive report. Do not infer
implementation or execution from this plan. Completing Phase 7B does not
retroactively mark Phase 7A fully validated.
