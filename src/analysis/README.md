# Phase 7A canonical branch analysis

This package implements the reviewed single-branch, leave-one-out, CKA and
conditional analyses. All experiment infrastructure lives here; tests are under
tests/analysis. Real execution is targeted at the verified Lumid sandbox.
Local tests use synthetic heads and read-only source/checkpoint replay.

## Selected-variant extension (Phase 7C)

The same pipeline supports one explicitly selected framework per task. Pass
`--profiles <json>` before the subcommand; omitting it preserves canonical
H0's branches, matrix and output paths. Supported families are `canonical`
(`H0`), `temporal` (Phase 6), `lstm_capacity` (Phase 6.5A), and
`residual_cnn` (Phase 6.5D, price only). Configuration names come directly
from the existing feature registries, not a second copied catalogue.

The owner-approved roster is encoded in
[`configs/phase7c_selected_profiles.json`](configs/phase7c_selected_profiles.json):
classification **HC-ST**, price **HB-AL**, and realised variance **HC-AL**,
fixed across both walks. These use equal-walk epoch-50 macro-F1 (classification)
or MAE (regression), not pooled-row weighting. The variance choice is the best
noncanonical, non-duplicate variant by that MAE rule; H0 has lower MAE.

Use the explicit `--phase7c` shortcut (canonical H0 stays the default):

```bash
PYTHONPATH=src .venv/bin/python3 -m analysis --phase7c audit
PYTHONPATH=src .venv/bin/python3 -m analysis --phase7c launch
```

For the owner-approved dev4 numerical amendment, additionally pass
`--tolerances src/analysis/configs/phase7c_replay_policy.json` and
`--output-root experiments/phase7/selected_variants/dev4` before the
subcommand. It permits at most `0.001` absolute auxiliary median Rank IC
drift (plus the unchanged relative term), without changing primary metric,
raw-output or integrity tolerances. The original failed prelaunch audit and
policy stay intact in the parent output root. See the Phase 7C plan for the
disclosed motivation and approval; do not apply this to archived Phase 7A.

The selected matrix has 68 standalone/removal heads, six existing full-model
references, 222 total snapshots and twelve CKA matrices. Nothing is launched
by adding the roster; the commands above do not train models.

Custom rosters still use the task-keyed `--profiles <json>` schema. For example,
the approved roster has the following structure (full reasons are in the file):

```json
{
  "profiles": {
    "classification_h2": {"family": "temporal", "configuration": "HC-ST", "selection_reason": "Highest equal-walk mean epoch-50 macro-F1"},
    "absolute_price_h8": {"family": "temporal", "configuration": "HB-AL", "selection_reason": "Lowest equal-walk mean epoch-50 price MAE"},
    "realised_variance": {"family": "temporal", "configuration": "HC-AL", "selection_reason": "Lowest equal-walk mean epoch-50 MAE among noncanonical, non-duplicate variants; H0 remains better"}
  }
}
```

Select exactly one configuration per task, with the same configuration in
both walks. A variant can serve multiple tasks; do not expand a three-by-three
cross-product. Record the selection rationale before analysis; there is no
automatic ranking/selection. Freeze the profiles in each output root.

```bash
PYTHONPATH=src .venv/bin/python3 -m analysis --profiles <json> audit
PYTHONPATH=src .venv/bin/python3 -m analysis --profiles <json> launch
```

`launch` remains preparation/manifest-only without `--execute`. Production
still requires verified Lumid admission, synchronized dependencies and
explicit execution. The approved dev4 run finished all 68 heads and twelve
replay-valid CKA matrices. Its
[archived report](../../experiments/phase7/selected_variants/dev4/reports/selected_variants_seed0/summary.md)
remains partial at 220/222 validated snapshots: price A-B/Walk 1/epoch 50
and A-noS/Walk 2/epoch 5 retain numerical replay exceptions. No further
tolerance waiver was applied; see the Phase 7C plan for details.
Do not reuse an old Phase 7A output root with changed code/profiles.

Selected-variant outputs default to `experiments/phase7/selected_variants/`.
Branch maps, standalone/removal counts, source paths, full-model references,
snapshot counts and plots are derived from the profiles. Five branches give
ten probes per task/walk; six branches give twelve. CKA uses each task's own
training/evaluation populations separately, giving twelve matrices for the
three-task/two-walk scope. The lean selected-variant completion gate requires
standalone/removal results and CKA, not conditional tables or SHAP.

Native master stores are anchored to the canonical ordered task rows and
targets, with explicit path relocation, file/array/encoder hashes, unchanged
canonical blocks and training-only scaler parity. Original full-model
checkpoints are replayed, never retrained. Existing runs are not silently
relabeled or reused across profiles merely because a branch has the same
name; source/configuration/code provenance must match. Original code hashes
in archived Phase 7A artifacts are not rewritten by this refactor.

See [the Phase 7C plan](../../docs/phase_plan/2026-10-18-phase-7c-selected-variant-analysis-plan.md).

## Workflow

From the persistent sandbox checkout, use its verified Python environment:

~~~bash
PYTHONPATH=src .venv/bin/python3 -m analysis --help
PYTHONPATH=src .venv/bin/python3 -m analysis --sandbox <actual-hostname> --site <site> audit
PYTHONPATH=src .venv/bin/python3 -m analysis prepare
PYTHONPATH=src .venv/bin/python3 -m analysis validate --stage preparation
PYTHONPATH=src .venv/bin/python3 -m analysis similarity
PYTHONPATH=src .venv/bin/python3 -m analysis --sandbox <actual-hostname> --site <site> launch --execute
PYTHONPATH=src .venv/bin/python3 -m analysis validate --stage runs
PYTHONPATH=src .venv/bin/python3 -m analysis conditional
PYTHONPATH=src .venv/bin/python3 -m analysis report
~~~

Launch without --execute writes preparation/manifests only. Full real-head
training requires --execute, the verified sandbox hostname/site, persistent
/home paths, CUDA, a clean committed checkout, and successful bounded preflight.
The run command coordinates all stages; run --execute --detach records a PID
and combined log under the output root. Logs do not prevent sandbox reclaim.

Global options --repo-root and --output-root select paths. --path-map accepts
a JSON list with recorded_root/resolved_root entries for legacy absolute paths.
Original input manifests are never rewritten. Dependency inventory follows
hash-backed references, resolving short filenames relative to their containing
manifest. The complete current real-source inventory contains 155 files.

## Results and failure handling

The registry contains 60 new seed-0 runs across three tasks and two walks, plus
six immutable H0 references. Each run saves evaluation checkpoints at epochs
5/15/50; epoch 50 is primary. A failed head restarts at epoch 1 on the next
explicit launch. Its partial attempt is archived, and verified completed runs
are reused. No optimizer/RNG training-resume API is present.

Exact hashes, source/target identities, branch maps and finite values remain
required. ReplayPolicy declares separate absolute tolerances for logits,
probabilities, RV outputs, ordinary metrics, rank metrics and CKA, alongside
relative tolerance. Pass --tolerances with a ReplayPolicy JSON to override
defaults before preparation. The first manifest freezes the policy; later
changes are rejected. Accepted drift is visibly logged and stored with
measured errors and thresholds. Invalid entries are isolated; other entries
continue. No invalid matrix is labelled complete. report --allow-partial
produces an explicitly incomplete report with omissions.

Outputs default to experiments/phase7/ablation: manifests, feature-view/scaler
records, downstream checkpoints/predictions, numerical analyses, and
reports/canonical_five_branch_seed0 containing JSON/CSV, PNG/SVG and summary.md.
Existing input files and H0 artifacts remain read-only.

## Modules

types/contracts define shared records and fixed recipes. runtime/sources
handle environment, dependencies and immutable source data. features/tasks
prepare coordinates and reuse original heads/losses. training owns fresh-head
fitting. ablation/similarity/conditional implement the scientific analyses.
artifacts/validation/replay handle persistence, diagnostics and verification.
pipeline coordinates stages; cli only dispatches commands. reporting exports
validated results and includes warning/incompleteness status.

Scope excludes prediction-error overlap, resource comparisons, extra seeds,
pairwise heads and probability/imputation subgroup tables. Bootstrap uncertainty
remains optional future work. Effects are descriptive, not significance or
equivalence claims.

## Development checks

~~~bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/phase7-matplotlib \
  PYTHONPATH=src .venv/bin/python3 -m unittest discover -s tests/analysis -v
~~~

The synthetic training test explicitly mocks production admission and never
uses real task rows. A local passing suite does not admit the Lumid runtime
or mean the 60-run matrix has executed.
