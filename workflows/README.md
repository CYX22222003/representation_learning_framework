# Lumid FlowMesh evidence workflows

These workflows run as non-interactive FlowMesh Python tasks. They do not use
or inherit a Lumid Sandbox. The versioned image supplies the code, dependencies,
and a bounded read-only evidence bundle containing scalar metrics and reports;
it does not contain datasets, feature stores, predictions, or checkpoints.

For a local xLSTM-Mixer dependency/CUDA compatibility check, use the optional
`xlstm-compat` image target documented in
[`docker/README.xlstm-mixer.md`](../docker/README.xlstm-mixer.md). Its synthetic
smoke script can also run directly in a new sandbox.

## Workflows

The single-DAG demonstration is documented in
[`research_audit_demonstration_plan.md`](research_audit_demonstration_plan.md).
It defines six major stages and 19 audit nodes, with separate encoder-variant,
external-representation, and task-specific benchmark groups. The new combined
workflow uses a separate standard-library audit image and a locked report bundle.
Each of its 19 nodes requests one GPU of any model, one CPU core, and 2 GiB of
memory. The audit code performs report inspection on CPU; the GPU request
controls worker allocation. The two-node handoff smoke remains a CPU probe.

| File | Purpose |
|---|---|
| `flowmesh_research_audit.yaml` | Complete 19-node evidence DAG, scientific metrics, CKA/resources, and matched synthesis. |
| `flowmesh_audit_dag_smoke.yaml` | Two-node structured-output and artifact handoff probe using the public Python base image. |
| `research_audit_evidence_manifest.json` | Exact evidence inventory, file/adapter hashes, dependencies, and declared metrics for the combined DAG. |
| `flowmesh_image_smoke.yaml` | Verify the v0.1.1 runtime image and imports. |
| `flowmesh_encoder_pretraining_audit.yaml` | Inspect six canonical encoder trajectories. |
| `flowmesh_downstream_tasks_audit.yaml` | Inspect H0 on three tasks and two walks. |
| `flowmesh_encoder_variants_audit.yaml` | Verify the 66-entry variant matrix and report hashes. |
| `flowmesh_baseline_comparison_audit.yaml` | Assemble H0, internal-envelope, and three external baselines. |

The four earlier standalone evidence workflows require image tag `flowmesh-v0.2.0`. Because the
Harbor project is public for anonymous FlowMesh pulls, review the bounded
experiment reports before publishing the image.

## Combined research audit: local preparation and verification

Run from the repository root using the WSL environment. All experiment inputs
are read-only. The bundle contains JSON/CSV/Markdown evidence, with no arrays,
weights, prediction files, or credentials. Adapter checks execute in each node;
source/feature/checkpoint replay remains explicitly labelled as prior evidence.

Reconstruct the locked bundle once; preparation refuses to replace an existing
bundle. If it already exists, proceed to validation.

```bash
.venv/bin/python3 -B scripts/prepare_research_audit_bundle.py \
  --frozen-manifest workflows/research_audit_evidence_manifest.json

.venv/bin/python3 -B scripts/generate_research_audit_workflow.py --check
.venv/bin/python3 -B scripts/validate_research_audit_workflow.py

.venv/bin/python3 -B scripts/run_research_audit.py \
  --output-dir build/flowmesh-research-audit-runs/local-demo

.venv/bin/python3 -B -m unittest discover \
  -s tests/platform_integration -p 'test_research_audit.py' -v
```

The local runner executes the exact YAML Python entry points, passes upstream
results along the declared edges, and enforces emitted-metric and output-size
limits. It writes one JSON per node and
`research_synthesis/research_report.md`. Every output identifies its evidence
bundle, population, principal epoch, recorded versus executed checks, and
upstream receipts. Scientific metrics use node/method/protocol/task/walk names
to prevent collisions. Epochs 5/15 are supporting evidence; epoch 50 supplies
the scalar scientific metrics. Non-learned references are included; undefined
correlations remain null and are not emitted as fabricated zeros.

The synthesis includes descriptive candidate-minus-reference differences
against H0, relevant predecessor/duplicate controls, and task-specific learned
controls. TA P2 stays on its matched intersection; P1U remains separate.
Frozen-representation and complete-system comparisons retain their different
interpretations. No training, inference, test-based model selection, or
cross-task ranking occurs.

To freeze an updated evidence version, use `prepare_research_audit_bundle.py`
without `--frozen-manifest`, specify a **new** `--output-dir` and
`--manifest-output`, then regenerate the YAML against that bundle. The lock
preserves its original source revision and exact adapter hashes even when Git
HEAD later advances. Changed adapters or source files require a new evidence
version, rather than silently altering this workflow's inputs.

## Combined audit image and platform validation

The combined DAG references
`harbor.lum.id/cyx-fyp/representation-learning-framework:research-audit-v0.3.0`.
The image is defined by `docker/Dockerfile.research-audit` and contains only the
two audit modules and the explicit `build/flowmesh-research-evidence` bundle.
Its build gate checks bundle/adapter/file integrity. Worker Python needs only
the standard library; local YAML tooling uses the existing PyYAML installation.

The built local image has also been verified directly: all 19 audit nodes pass
with its filesystem read-only and networking disabled. Its 915 packaged
evidence files match the frozen inventory, producing 260 normalized result
rows and 212 matched comparisons without a host evidence mount. The prepared
`build/flowmesh-research-evidence/` directory is copied to `/app/evidence` at
build time. On 2026-10-10, the owner supplied the combined DAG screenshot and
reported that all 19 stages passed on FlowMesh. Platform metrics and output
artifacts have not yet been independently inspected; anonymous registry pulls
remain unverified.

```bash
docker build --platform=linux/amd64 \
  -f docker/Dockerfile.research-audit \
  -t harbor.lum.id/cyx-fyp/representation-learning-framework:research-audit-v0.3.0 .

.venv/bin/python3 -B scripts/validate_research_audit_workflow.py \
  --smoke --workflow workflows/flowmesh_audit_dag_smoke.yaml --platform

.venv/bin/python3 -B scripts/validate_research_audit_workflow.py --platform
```

Platform validation reads `LUMID_PAT` from the environment and calls only the
validation endpoint. It does not dispatch a task. A successful response checks
the compiled node roster and dependency edges where the platform returns them.
The GPU resource configuration has been checked locally against the public
Studio editor's fields. The owner subsequently reported successful execution
of the full DAG; the agent did not capture a separate platform validation
receipt for that configuration. Platform artifact mounts and returned metric
values still need direct inspection. For future runs, publish the built image
to a registry accessible to FlowMesh workers. These scripts do not publish
images or submit jobs.

Validation status and remaining execution checks are recorded in the
[implementation status](research_audit_demonstration_plan.md#implementation-status).

## Earlier standalone workflows: build and publish

```bash
docker build \
  --platform=linux/amd64 \
  -f docker/Dockerfile \
  -t harbor.lum.id/cyx-fyp/representation-learning-framework:flowmesh-v0.2.0 \
  .

docker push \
  harbor.lum.id/cyx-fyp/representation-learning-framework:flowmesh-v0.2.0
```

## Validate and submit one workflow

Set `WORKFLOW_FILE` to one of the four audit YAML files.

```bash
WORKFLOW_FILE=workflows/flowmesh_encoder_pretraining_audit.yaml

curl -sS -X POST \
  https://lum.id/fm/home/api/v1/workflows/validate \
  -H "Authorization: Bearer $LUMID_PAT" \
  -H "Content-Type: text/plain" \
  --data-binary "@$WORKFLOW_FILE" \
  | jq .

curl -sS -X POST \
  https://lum.id/fm/home/api/v1/workflows \
  -H "Authorization: Bearer $LUMID_PAT" \
  -H "Content-Type: text/plain" \
  --data-binary "@$WORKFLOW_FILE" \
  | jq .
```

Poll the returned task id and retrieve its returned output:

```bash
TASK_ID=tsk-replace-me

curl -sS \
  -H "Authorization: Bearer $LUMID_PAT" \
  "https://lum.id/fm/home/api/v1/tasks/$TASK_ID" \
  | jq '{task_id, workflow_id, status, completed, failed, error}'

curl -sS \
  -H "Authorization: Bearer $LUMID_PAT" \
  "https://lum.id/fm/home/api/v1/results/$TASK_ID" \
  | jq .
```

Each task also writes a named JSON audit under `FLOWMESH_OUTPUT`. Download the
result and artifact bundle with:

```bash
curl -sS \
  -H "Authorization: Bearer $LUMID_PAT" \
  "https://lum.id/fm/home/api/v1/results/$TASK_ID/bundle?include=results&include=artifacts" \
  --output "$TASK_ID.tar.gz"
```
