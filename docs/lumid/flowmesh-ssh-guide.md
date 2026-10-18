# FlowMesh SSH tasks

Read this only for interactive worker debugging or an SSH-task image. For
normal Python/DAG execution, use the [Workflow/FlowMesh guide](workflow-flowmesh-guide.md).
This is a concise 2026-10-18 consolidation of the supplied SSH guide, whose
examples were exercised on 2026-08-21; it does not claim a new live session test.
The source snapshot hash is recorded in that guide's [source inventory](workflow-flowmesh-guide.md#sources).

## Task and API

```yaml
apiVersion: flowmesh/v1
kind: Task
metadata:
  name: debug-shell
spec:
  taskType: ssh
  accessMode: forward
  ttlSeconds: 900
  idleTimeoutSeconds: 600
  authorizedKeys:
    - "ssh-ed25519 AAAA... replace-with-your-public-key"
```

Supply only a public key. The supplied guide's default SSH image provides
`sshd`; leave `image` unset unless a custom image preserves that contract.
`ttlSeconds` is a hard lifetime; `idleTimeoutSeconds` measures inactivity.
Neither creates persistent worker storage.

Use the [API setup](workflow-flowmesh-guide.md#3-validate-submit-inspect-and-stop-through-the-api):
validate and submit the YAML, record `tasks[].task_id`, and poll
`GET /tasks/<task_id>` on the same site. Read logs at
`GET /tasks/<task_id>/logs`. For forward mode, wait for returned `host` and
`port`. Set `SSH_HOST` and `SSH_PORT` to those returned values, then connect as
the source default user `flowmesh`:

```bash
ssh -o ConnectionAttempts=3 -p "$SSH_PORT" "flowmesh@$SSH_HOST"
```

`directHost`/`directPort` are worker-local hints and may be unreachable from a
laptop. Do not substitute a sandbox gateway port. Re-read live hints if a task
stopped or expired. Record/verify the session host key; do not routinely disable
host-key checks to work around a recreated container.

| `accessMode` | Connection and constraints |
|---|---|
| `forward` | Returned public TCP host/port; normal SSH/SCP/rsync/forwarding |
| `proxy` | WebSocket SSH stream at `wss://lum.id/fm/<site>/api/v1/ssh/tasks/<task_id>/proxy` with Bearer auth; needs a stdio/WebSocket `ProxyCommand` bridge |
| `direct` | Worker-local address; requires appropriate LAN access |

The source reports finite forward pools: cloud `32032–32063`, office
`32640–32655`, home `32672–32687`. Treat these as dated diagnostics, not ports
to select manually; always use task hints. Proxy mode consumes no forward port.

## Image and storage contract

- Worker files are ephemeral. The supplied guide describes `/huggingface` as
  a node-local cache, no sandbox `/datasets`, and no persistent `/home/<user>`.
  Check actual mounts and export/download results before stopping the session.
- Default SSH images were minimal and unprivileged (`flowmesh`, no sudo), with
  no Python, nvcc, or curl guaranteed. Install required OS/Python packages at
  image-build time; do not assume sandbox root access.
- For custom SSH images, inherit the appropriate versioned/digest-pinned
  `ghcr.io/mlsys-io/flowmesh_ssh:<version>-cpu` or `-gpu` base. Preserve its
  `sshd`, `SSH_USER`/`AUTHORIZED_KEYS` handling, entrypoint, and final login
  user. A generic Ubuntu/Python image can exit before SSH becomes ready.
- `latest-cpu`/`latest-gpu` appeared in source examples; resolve and pin a tested
  version/digest for a reproducible build. Use `linux/amd64`, check worker pull
  permissions, and review public Harbor image contents before publishing.
- The source observed 64 MB `/dev/shm` in SSH tasks and DataLoader SIGBUS with
  multiple workers. Check `df -h /dev/shm`; use `num_workers=0` for a bounded
  probe until adequate shared memory is established. This observation does not
  set a blanket policy for Python tasks or sandboxes.
- On Blackwell hardware, test a CUDA kernel as well as
  `torch.cuda.is_available()`. The source reports older cu121/cu124 wheels
  failing with `no kernel image is available`; select a stack supporting the
  actual GPU architecture and validate it on the target.

## Stop and diagnose

When the user requests stopping the session, retain required outputs and call
`POST /tasks/<task_id>/stop` on its site API. Inspect status afterward. No
sandbox DELETE is involved.

| Symptom | Likely checks from the supplied guide |
|---|---|
| Container exited before SSH ready | Missing sshd or incompatible entrypoint/user in custom image |
| No worker satisfies requirements | Wrong site, missing capability, excessive resources |
| TCP opens then key exchange closes | Stale hint, TTL/idle expiry, or a newly bound relay not yet healthy |
| No public `host`/`port` | Task readiness or exhausted forward pool; proxy mode may be appropriate |
| WebSocket `403` | Can represent auth, unknown/not-ready task, or relay failure; inspect task state/logs |
| Image pull error | Missing tag, private anonymous-pull project, or incompatible architecture |

Do not infer a current root cause from the status code alone. Preserve the task
ID, site, image identity, request/response, and logs without credentials.
