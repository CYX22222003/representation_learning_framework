# Lumid Workflow and FlowMesh

Repository reference, consolidated **2026-10-18** from the owner-supplied
platform guides listed under [Sources](#sources). This replaces the workflow
skill's dependency on guide files outside the checkout. It keeps execution,
API, image, artifact, and measurement contracts; long Studio walkthroughs,
screenshots, and unrelated worked-study results are omitted.

These are maintained summaries of supplied documentation and repository
examples. They do not establish fresh platform validation, image publication,
or successful execution. Recheck live schemas, permissions, and site capacity
when performing an actual deployment.

## 1. Select the runtime

| Component | Use and boundary |
|---|---|
| Sandbox | Development with persistent `/home/<user>`; see the [sandbox guide](lumid_container_guide.md) |
| FlowMesh Python task | Non-interactive code in an ephemeral worker container |
| FlowMesh graph | Tasks with explicit dependencies and per-node resources |
| FlowMesh SSH task | Temporary interactive worker debugging; see the [SSH guide](flowmesh-ssh-guide.md) |
| Lumilake `ops` / `LambdaOp` | A different workflow dialect; observed import restrictions make it unsuitable for this project's normal ML runtime |
| OCI image / Harbor | Code and runtime dependencies / registry holding the image |

Workers do not inherit a sandbox's checkout, environment, persistent home, or
`/datasets`. Supply each input through the image, a supported workflow input
or mount, or an explicit service query. The source SSH guide describes
`/huggingface` as a node-local model cache; check the selected task's mounts
before relying on it. Write important outputs to the task's artifact location
or external storage before the container ends.

## 2. YAML and Python task contract

Use the FlowMesh envelope, not a top-level `tasks:` list or Lumilake `ops`:

```yaml
apiVersion: flowmesh/v1
kind: Task
metadata:
  name: python-smoke
spec:
  taskType: python
  image: python:3.12.3-slim-bookworm
  entrypoint: main
  emits: [smoke_pass]
  timeoutSeconds: 60
  network: none
  resources:
    hardware: {cpu: 1, memory: 512Mi}
  pythonOutput:
    maxBytes: 1048576
  code: |
    def main():
        return {"output": {"message": "ok"}, "metrics": {"smoke_pass": 1.0}}
```

For a DAG, the repository uses `spec.graph.nodes[]`. Each node has a `name`,
`dependsOn: [upstream_name]`, and its own task `spec`. The source platform guide
also describes `spec.stages`; retain an existing dialect instead of silently
rewriting it. See the [two-node handoff example](../../workflows/flowmesh_audit_dag_smoke.yaml)
for exact structured-output and file-artifact wiring.

| Concern | Contract |
|---|---|
| Entry function | `entrypoint: main` selects the callable; a zero-argument function receives no arguments |
| Connected values | `def main(prepare)` receives the named upstream output; `def main(inputs)` receives all connected steps |
| Connected artifacts | In the `inputs` form, each connected step exposes `output` and `artifacts`; the repository smoke handles dict and attribute wrappers |
| Metrics | Return `{"metrics": {...}}`; every name in `emits` must be returned or the step fails |
| Files | Write artifact files beneath `FLOWMESH_OUTPUT`; persist/download them before worker cleanup |
| Network | Source default is `none`; package installation and HTTP queries need an enabled supported network such as `bridge` |
| Dependencies | Bake pinned packages into a versioned image, or declare requirements through the supported task fields; do not assume host packages |
| CPU/RAM | Use `resources.hardware.cpu` and `memory`; container limits differ from host totals |
| GPU | Repository examples use `resources.hardware.gpu: {type: any, count: 1}`; inspect site capacity before requesting it |
| Timeout/output size | Declare `timeoutSeconds` and suitable `pythonOutput.maxBytes`; large checkpoints belong in artifact storage, not inline JSON |

The source guide's Python default image is `python:3.12-slim`. Pin an explicit
version/digest for a reproducible run. Python tasks run with restricted runtime
privileges; install OS packages when building the image. Preserve the image's
bootstrap-compatible default user: this repository's non-interactive images
leave `USER` unset so FlowMesh can prepare the mounted output directory before
dropping privileges. SSH images have a separate user/entrypoint contract.

## 3. Validate, submit, inspect, and stop through the API

For `home`, set `FM_API=https://lum.id/fm/home/api/v1`; `office` has the same
shape with its site name. Use the same site for validation, submission, polling,
results, and stopping. Job/task IDs alone do not identify a site.

| Operation | Request relative to `FM_API` |
|---|---|
| Validate without worker dispatch | `POST /workflows/validate`; YAML body, `Content-Type: text/plain` |
| Submit and allocate workers | `POST /workflows`; same YAML body |
| Inspect task state | `GET /tasks/<task_id>` |
| Read logs | `GET /tasks/<task_id>/logs` |
| Retrieve returned results | `GET /results/<task_id>` |
| Download results and artifacts | `GET /results/<task_id>/bundle?include=results&include=artifacts` |
| Stop a requested task/session | `POST /tasks/<task_id>/stop` |

The source SSH guide uses `GET https://lum.id/fm/api/v1/nodes` for cross-site
node discovery and an unqualified `/fm/api/v1` cloud route. The repository
validator constructs `/fm/cloud/api/v1` when `--site cloud` is passed. Confirm
the deployment's cloud route before relying on either form; this consolidation
does not resolve that source difference through a live cloud test.

### Credentials and shell setup

Use the owner-selected credential file described in the
[sandbox API credential section](lumid_container_guide.md#52-credentials-and-a-reusable-python-client):
`/home/guest1/.config/lumid/credentials.env`, variable `LUMID_PAT`. Its path is
workstation-specific; the token stays outside Git. Sandbox write permissions
do not establish FlowMesh permissions. A `401`/`403` requires checking the
current credential and operation's permissions rather than guessing a scope.

For this trusted shell-assignment file, in Bash:

```bash
set -a
. /home/guest1/.config/lumid/credentials.env
set +a
FM_API=https://lum.id/fm/home/api/v1
WORKFLOW_FILE=workflows/flowmesh_audit_dag_smoke.yaml

# Pass the PAT header on stdin rather than in curl's process arguments.
# Do not enable shell tracing while handling credentials.
fm_api() {
  printf 'Authorization: Bearer %s\n' "$LUMID_PAT" |
    curl --silent --show-error --fail-with-body --max-time 30 --header @- "$@"
}
```

These commands do not follow redirects. The function uses stdin for headers;
send workflow content from a file via `--data-binary`, not from stdin.

Validate first and inspect the response for `valid: true` or `ok: true`:

```bash
fm_api --request POST --header 'Content-Type: text/plain' \
  --data-binary "@$WORKFLOW_FILE" "$FM_API/workflows/validate"
```

For the repository's two-node smoke, this existing command also checks the
local graph/code and can request platform validation without dispatching:

```bash
.venv/bin/python3 -B scripts/validate_research_audit_workflow.py \
  --smoke --workflow workflows/flowmesh_audit_dag_smoke.yaml --platform --site home
```

After successful validation and when execution is requested, submit once:

```bash
fm_api --request POST --header 'Content-Type: text/plain' \
  --data-binary "@$WORKFLOW_FILE" "$FM_API/workflows"
```

Record the returned `workflow_id` and every `tasks[].task_id`. Poll the task IDs
returned by the API rather than substituting a workflow ID:

```bash
TASK_ID=tsk-replace-me
fm_api "$FM_API/tasks/$TASK_ID"
fm_api "$FM_API/tasks/$TASK_ID/logs"
fm_api "$FM_API/results/$TASK_ID"
fm_api --max-time 120 --output "$TASK_ID.tar.gz" \
  "$FM_API/results/$TASK_ID/bundle?include=results&include=artifacts"
```

Poll at a reasonable interval, such as ten seconds. Treat HTTP failures as
unavailable status; label retained snapshots stale. A task's success state
needs returned outputs, declared metrics, and artifact checks before claiming
the requested experiment completed. After a submission timeout, inspect
available job records before resubmitting; these examples do not implement
automatic POST retries.

Stop only the exact task the user requested, retaining required outputs first:

```bash
fm_api --request POST "$FM_API/tasks/$TASK_ID/stop"
```

The source workflow guide also describes Studio fleet jobs at
`/api/v1/me/fleet/jobs` and legacy compute ownership/status surfaces under
`/me/compute/jobs/:site/:job_id`. Those are separate APIs and identifier spaces;
do not prepend `FM_API` to them or assume they accept the same request body.
Its legacy compute `progress` describes job-level phases, not per-op state;
`batch_progress.completed` is a count. A wrong site or unowned compute job can
return `404`. Use the raw FlowMesh surface above for these task examples.

## 4. Images, Harbor, data, and artifacts

1. Inspect the selected stage's imports and system libraries. Pin its Python,
   PyTorch/CUDA, packages, source commit, image tag, and preferably image digest.
   Build `linux/amd64` for the documented workers; do not copy `.venv`.
2. Keep the build context bounded with `.dockerignore`. Include only the
   documented data/checkpoint subset needed by the stage. Credentials, private
   keys, `.env` files, unrelated datasets, and broad experiment trees stay out.
3. The supplied SSH platform guide describes anonymous worker image pulls and
   therefore public Harbor projects. Verify current pull access before use;
   sandbox private-image credentials do not establish worker access. Review
   source/data visibility before a requested publication.
4. Harbor OIDC login uses a CLI secret. Use interactive `docker login
   harbor.lum.id -u <user>` or a supported password-stdin flow; keep secret
   values out of command arguments and the image. Push only when requested.
5. A custom image is useful for heavy stacks, OS packages, CUDA, or repeat jobs.
   Normal Python nodes do not need an SSH server. For interactive SSH tasks,
   inherit an SSH-ready FlowMesh image and retain its login/entrypoint contract.
6. Query FinData through the worker's documented site-specific service endpoint
   with the required runtime credentials/network. Stage exact frozen query
   results when reproducibility requires them; neither sandbox attachments nor
   sandbox `/datasets` are automatically available to a worker.
7. Verify upstream artifact handoff and persistence/download of checkpoints,
   predictions, logs, metrics, and reports. A successful container exit alone
   does not prove that those files survived.

See [workflow commands and image evidence](../../workflows/README.md),
[the two-node smoke](../../workflows/flowmesh_audit_dag_smoke.yaml),
[the combined evidence DAG](../../workflows/flowmesh_research_audit.yaml), and
[the bounded audit image](../../docker/Dockerfile.research-audit).
The repository's audit workflows inspect frozen reports; requesting a GPU for
placement in an audit is not encoder training, inference, or fresh replay.
Keep image build, local validation, platform validation, publication, execution,
and independent result inspection as separately evidenced steps.

## 5. Studies and scheduled comparisons, when needed

These are platform study semantics from the supplied guide, not additional
requirements for every workflow or changes to the project's frozen ML protocol.

- A study adds a declared metric key and dataset/case scope to a workflow.
  Match emitted metric names exactly. A missing metric is not a zero, and
  a saturated feasibility gate is not a useful comparison metric.
- Study definition is queued as an intent; inspect its applied state. Adding
  an experiment uses a separate operation; repeating an experiment ID replaces
  that experiment. Prefer a fork when changing the meaning of a running study.
- `compare_within` dimensions identify a common measurement instrument.
  `dataset_version` fences result cohorts without deleting earlier rows.
  Preserve common populations and measurement settings before comparing arms.
- `success_criteria` permits `n`, `best_n`, `best_mean`, `baseline_n`,
  `baseline_mean`, `baseline_value`, `delta`, `delta_pp`, and
  `mean_<arm_id>`. The supplied guide disallows function calls such as `abs()`.
- Scheduled fires pass no arguments. `rotate_args` can select a row from a
  frozen dataset; `engine.arms_concurrency: 1` serializes arms when simultaneous
  models would exceed a worker's GPU memory. Do not infer VRAM reservation
  across otherwise concurrent jobs.
- Platform verdicts can be withheld for incompatible instruments or uncertainty.
  Inspect emitted metric counts, scope, and comparisons rather than treating a
  UI verdict as proof of scientific superiority.

## 6. Troubleshooting boundaries

| Symptom | Check |
|---|---|
| Wrong dialect / validation refusal | `flowmesh/v1`, task envelope, graph/node field names, declared dependency names |
| Import or pip failure | Image packages, task privileges, and enabled network for runtime installation |
| No eligible worker | Site, node inventory, CPU/RAM/GPU requirements, image architecture |
| Image pull failure | Tag/digest exists, anonymous/private pull access, `linux/amd64` |
| Missing declared metric | Exact `emits` name and return `metrics` dictionary |
| Missing upstream file | Dependency edge, connected artifact path, upstream `FLOWMESH_OUTPUT` |
| Structured-output truncation | `pythonOutput.maxBytes`; move large outputs to files |
| Generic `LambdaOp` failure | Supplied bug report reproduced failures on `import math`/`import os`; use a normal Python task for imports |

The LambdaOp report also required the function definition as the first binding;
moving an import into the function passed compilation but still failed at
runtime. These are source observations, not fresh tests or universal current
Python limitations. The source workflow guide describes a newer isolated
per-row Python step; do not conflate that step with the old `LambdaOp` path.

## Sources

Consolidated from the supplied local snapshots below; the originals are no
longer runtime/document dependencies. UI screenshots, example-study numbers,
and long changelogs were excluded from this operational summary.

| Source snapshot | SHA-256 |
|---|---|
| `lumid_workflow_guide.md` (changelog through 2026-09-27) | `251126c1744d57187fbcf0a6f7f95e759ea8ce4a6be011cb6807981a2070f2b2` |
| `lumid-flowmesh-work-container-platform-guide.md` | `adfe9f780644fdc1e9110b863aa936d7fce5bd5de3d8495c882b00fe1be28a88` |
| `lumid-flowmesh-ssh.md` (examples exercised 2026-08-21) | `addd963e78c97c12a35490881cbcc088c5c9ed9fd0bcb319b718556f45aaf750` |
| `lumid_python_workflow_bug_reports.md` | `48cb0d766872c2895ebec59212b9db92ba4f5629fb604d08241a9884ce30f1c9` |

Official platform entry points for checking updated contracts:
[compute](https://lum.id/studio/docs/compute),
[FlowMesh SSH](https://lum.id/studio/docs/flowmesh-ssh), and
[operations](https://lum.id/studio/docs/operations).
