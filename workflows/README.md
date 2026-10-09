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

| File | Purpose |
|---|---|
| `flowmesh_image_smoke.yaml` | Verify the v0.1.1 runtime image and imports. |
| `flowmesh_encoder_pretraining_audit.yaml` | Inspect six canonical encoder trajectories. |
| `flowmesh_downstream_tasks_audit.yaml` | Inspect H0 on three tasks and two walks. |
| `flowmesh_encoder_variants_audit.yaml` | Verify the 66-entry variant matrix and report hashes. |
| `flowmesh_baseline_comparison_audit.yaml` | Assemble H0, internal-envelope, and three external baselines. |

The four evidence workflows require image tag `flowmesh-v0.2.0`. Because the
Harbor project is public for anonymous FlowMesh pulls, review the bounded
experiment reports before publishing the image.

## Build and publish the evidence image

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
