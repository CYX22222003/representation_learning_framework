# Phase 7A canonical branch analysis

This package implements the reviewed single-branch, leave-one-out, CKA and
conditional analyses. All experiment infrastructure lives here; tests are under
tests/analysis. Real execution is targeted at the verified Lumid sandbox.
Local tests use synthetic heads and read-only source/checkpoint replay.

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
