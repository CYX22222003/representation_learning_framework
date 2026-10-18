---
name: lumid-sandbox-ops
description: Operate Lumid Research Fleet sandboxes safely and reproducibly. Use when creating, connecting to, testing, running workloads in, exposing ports from, transferring files to, monitoring, or troubleshooting Lumid sandboxes.
---

# Lumid Sandbox Operations

Use this skill for day-to-day work with Lumid Research Fleet sandboxes.

The operating model is:

- **Sandbox = compute environment**
- **FinData / Lumid Data / LQT = remote data services**
- **`/datasets` = shared read-only file datasets**
- **`/home/<user>` = persistent user storage**
- **container-local paths such as `/workspace` and `/root` = ephemeral unless the platform says otherwise**
- **public ports = mappings from `lum.id:<public-port>` to a port listening inside the sandbox**

Prefer non-interactive SSH commands when the interactive shell is unavailable.

## 1. Core safety rules

1. Keep important code, checkpoints, and outputs under `/home/<user>` or another documented persistent store.
2. Treat `/workspace`, `/root`, installed packages, processes, and other container-local state as disposable.
3. Never upload or expose a private SSH key.
4. Do not assume a process is safely detached merely because the SSH connection closed.
5. For long-running jobs, use `nohup`, redirect stdin/stdout/stderr, and record the PID/log path.
6. A published port is Internet-reachable unless another access-control layer is documented. Add application authentication when needed.
7. Before trusting an image, verify the commands and libraries you actually need. Site-default images can be intentionally minimal.
8. Before a long unattended workload, verify the site's reclaim/TTL policy.

## 2. Site SSH gateways

Known site gateways:

```text
home   -> ssh -p 31223 gw@lum.id
office -> ssh -p 31226 gw@lum.id
```

The user connects to a **site gateway**, not directly to a sandbox IP.

The gateway identifies the user through the registered SSH public key and routes supported commands to the sandbox.

## 3. SSH key setup

### WSL/Linux

Check for an existing key:

```bash
ls -la ~/.ssh
```

Preferred key:

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

Print the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Check its fingerprint:

```bash
ssh-keygen -lf ~/.ssh/id_ed25519.pub
```

If no key exists:

```bash
ssh-keygen -t ed25519
```

Upload **only** the `.pub` key to Lumid Studio.

### Windows PowerShell

List local keys:

```powershell
Get-ChildItem $HOME\.ssh
```

Print the public key:

```powershell
Get-Content $HOME\.ssh\id_ed25519.pub
```

Fingerprint:

```powershell
ssh-keygen -lf $HOME\.ssh\id_ed25519.pub
```

Windows and WSL usually have different SSH keypairs. Register whichever keys will actually be used.

### Persistent GitHub identity inside the project sandbox

The current project sandbox may run commands with `$HOME=/root` even though
the persistent user volume is `/home/personai-korolev-tes`. A key placed in
`/root/.ssh` or a default `known_hosts` created there can disappear when the
sandbox is recreated. Keep the project GitHub identity and host record under
the persistent volume:

```text
/home/personai-korolev-tes/.ssh/id_ed25519
/home/personai-korolev-tes/.ssh/id_ed25519.pub
/home/personai-korolev-tes/.ssh/known_hosts
```

Expected permissions are `700` on `.ssh`, `600` on the private key and
`known_hosts`, and `644` on the public key. Never print, copy, commit, or
transfer the private key through experiment tooling.

Verify GitHub authentication explicitly:

```bash
ssh -i /home/personai-korolev-tes/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/home/personai-korolev-tes/.ssh/known_hosts \
  -o StrictHostKeyChecking=accept-new \
  -T git@github.com
```

GitHub prints a successful-authentication message and normally exits with
status `1` because it does not provide shell access. After the first verified
connection, configure the persistent repository rather than relying on the
container's default SSH search path:

```bash
cd /home/personai-korolev-tes/representation_learning_framework
git remote set-url origin \
  git@github.com:CYX22222003/representation_learning_framework.git
git config core.sshCommand \
  'ssh -i /home/personai-korolev-tes/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/personai-korolev-tes/.ssh/known_hosts -o StrictHostKeyChecking=yes'
git ls-remote origin HEAD
```

This repository-local configuration is stored beneath the persistent clone.
Revalidate it after every sandbox recreation before the first fetch or pull.

## 4. Diagnose SSH in layers

Use:

```bash
ssh -vvv -o ConnectTimeout=10 -p 31223 gw@lum.id
```

Interpret failures by layer.

### TCP timeout

Example:

```text
Connecting to lum.id [...] port 31223
Connection timed out
```

Meaning: the SSH gateway was not reached. Check:

```bash
ping lum.id
nc -vz -w 5 lum.id 31223
```

On Windows:

```powershell
Test-NetConnection lum.id -Port 31223
```

If another network works, the original Wi-Fi/firewall may be blocking the non-standard SSH port.

### Public-key rejection

If TCP/SSH negotiation succeeds but ends with:

```text
Permission denied (publickey,keyboard-interactive)
```

check that the offered key fingerprint matches the key registered in Lumid:

```bash
ssh -vvv -i ~/.ssh/id_ed25519 -p 31223 gw@lum.id
```

Look for:

```text
Offering public key: ...
Server accepts key: ...
```

### Interactive-shell failure

If authentication succeeds but no usable shell appears, do not assume the sandbox itself is unusable.

Test non-interactive execution:

```bash
ssh -p 31223 gw@lum.id 'hostname'
ssh -p 31223 gw@lum.id 'pwd'
ssh -p 31223 gw@lum.id 'whoami'
ssh -p 31223 gw@lum.id 'nvidia-smi'
```

A working remote command path can be sufficient for batch workloads.

## 5. Discover and target sandboxes

### 5.1 API access without Studio

Sandbox creation, status checks, quota inspection, and deletion can use HTTPS
directly. An existing PAT is sufficient; Studio does not need to be open.
The API contract was checked on **2026-10-18** against
<https://lum.id/sbx/openapi.json>. The schema uses `/api/...` paths; on this
deployment they are served beneath `https://lum.id/sbx`.

| Operation | Method | Full URL |
|---|---|---|
| Identify the caller | `GET` | `https://lum.id/sbx/api/whoami` |
| List the caller's sandboxes and site capacity | `GET` | `https://lum.id/sbx/api/sandboxes` |
| Inspect used and hard resource quota | `GET` | `https://lum.id/sbx/api/quota` |
| Create a sandbox | `POST` | `https://lum.id/sbx/api/sandboxes` |
| Delete one named sandbox | `DELETE` | `https://lum.id/sbx/api/sandboxes/<name>` |

The current schema has **no `GET /api/sandboxes/<name>` detail endpoint**.
Select the exact `name` from the list response. The list is per caller, not a
fleet-wide inventory. `GET /whoami` currently returns identity fields, not PAT
scopes; a successful read does not establish permission to create or delete.

### 5.2 Credentials and a reusable Python client

The owner-selected write-enabled credential file on this workstation is:

```text
/home/guest1/.config/lumid/credentials.env
```

It contains a `LUMID_PAT` assignment. Read that file explicitly for mutation
requests. During the 2026-10-18 session, the exported `LUMID_PAT` and
`/home/guest1/.lumid_env` held a different, read-only credential: deletion
returned `403`, while the PAT in `credentials.env` succeeded. These are dated
observations, not permanent statements about either token's permissions.

The observed write requirement is `sandbox:sandboxes:write` (or `*`). A `403`
reported that changing sandboxes requires that scope. If credentials need to
be issued or changed, the server points to
<https://lum.id/studio/account/tokens>; API operations themselves can then run
without Studio. Do not infer deletion permission from an earlier creation or
from another token with the same environment-variable name.

Keep credential contents outside Git and logs. Never print the PAT or put its
value in a command argument. The following standard-library helper reads plain
or quoted `LUMID_PAT=...` / `export LUMID_PAT=...` assignments without executing
the credential file. Replace the path for another workstation. Run this helper
and the subsequent Python examples in the **same script or Python session**.
From this repository, use `.venv/bin/python3`.

```python
import json
from pathlib import Path
import shlex
from urllib import error, parse, request

BASE = "https://lum.id/sbx/api"
CREDENTIALS = Path("/home/guest1/.config/lumid/credentials.env")

def load_pat(path):
    token = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == "LUMID_PAT":
            values = shlex.split(value, comments=True)
            if len(values) != 1 or not values[0]:
                raise ValueError("Unsupported or empty LUMID_PAT assignment")
            token = values[0]
    if not token:
        raise ValueError("LUMID_PAT is missing from the credential file")
    return token

class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

TOKEN = load_pat(CREDENTIALS)
opener = request.build_opener(NoRedirect())

def api(method, path, payload=None):
    headers = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = request.Request(BASE + path, data=body, headers=headers, method=method)
    try:
        with opener.open(req, timeout=30) as response:
            return response.status, json.load(response)
    except error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").replace(TOKEN, "[redacted]")
        raise RuntimeError(f"{method} {path}: HTTP {exc.code}: {detail}") from None
    except (error.URLError, OSError, ValueError) as exc:
        detail = str(exc).replace(TOKEN, "[redacted]")
        raise RuntimeError(f"{method} {path}: {detail}") from None

def show(value):
    print(json.dumps(value, indent=2).replace(TOKEN, "[redacted]"))
```

Redirects are disabled so an Authorization header is not forwarded elsewhere.
The helper does not automatically retry requests. After an uncertain POST or
DELETE outcome, inspect the sandbox list before deciding whether to retry.

### 5.3 Check capacity, quota, and the intended name

```python
show(api("GET", "/whoami")[1])
status, inventory = api("GET", "/sandboxes")
show(inventory)
show(api("GET", "/quota")[1])
```

The observed list response contains:

| JSON field | Interpretation |
|---|---|
| `sandboxes` | Caller-owned sandbox rows; select an exact `name` |
| `gpus_free` | Site-level free/total GPU capacity, e.g. `"2/5"` |
| `gpu` | GPU products, model/memory information, and per-sandbox limits |
| `cpu` | Advertised CPU sizes and general CPU/memory limits |
| `images` | Image catalog plus `default_cpu` and `default_gpu` |
| `data_sources` | Supported service IDs and connection descriptions |
| `ports` | Public-port pool availability and mapping limits |

Quota has `used` and `hard` mappings. Inspect `count/pods`, `requests.cpu`,
`requests.memory`, `limits.cpu`, `limits.memory`, and
`persistentvolumeclaims`. Terminating sandboxes can still occupy a pod slot.
Do not treat aggregate quota as an individual sandbox's original configuration.

### 5.4 Create a sandbox with explicit resources

The schema accepts these creation fields:

| Field | Meaning and schema limits |
|---|---|
| `name` | Sandbox identifier; schema default `"dev"`; choose an unused name |
| `image` | Optional image reference; choose from the inventory or an approved custom image |
| `cpu` | CPU cores, `1..32`; default `2` |
| `memory_gi` | System RAM in GiB, `1..128`; default `8` |
| `gpu` | GPU count, `0..2`; default `0`; site limits can be lower |
| `gpu_product` | Optional product selector; inspect the live product catalog |
| `ttl_hours` | Requested lifetime, `1..24`; default `8` |
| `data_sources` | Service-ID array; `[]` attaches no optional services |
| `ports` | Container-port array; `[]` requests no public application mappings |

Schema maxima are not a promise that a particular GPU node can fit the request.
On 2026-10-18 this site advertised at most one GPU per sandbox; requesting
32 GiB RAM returned `400` because the largest eligible GPU machine supported
29 GiB. An **8-CPU / 24-GiB / one-GPU** request succeeded. That allocation was
verified inside the container as `cpu.max = 800000 100000` and
`memory.max = 25769803776`. The visible GPU was an RTX PRO 4000 Blackwell SFF
with about 24 GB VRAM. System RAM and GPU VRAM are separate resources.

Choose CPU/RAM explicitly for the intended workload. The default 2 CPU / 8 GiB
allocation is not an experiment resource recommendation. Recheck current
capacity and scheduling errors; do not assume this dated hardware limit applies
to every Lumid site or future node.

Example for an authorized future creation; **the documentation itself does not
request a new sandbox**:

```python
name = "research-next"  # Replace with the intended unused name.
existing = api("GET", "/sandboxes")[1]["sandboxes"]
if any(row.get("name") == name for row in existing):
    raise RuntimeError(f"{name} already exists; no POST sent")

payload = {
    "name": name,
    "image": "pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime",
    "cpu": 8,
    "memory_gi": 24,
    "gpu": 1,
    "ttl_hours": 24,
    "data_sources": [],
    "ports": [],
}
status, created = api("POST", "/sandboxes", payload)
print("HTTP status:", status)
show(created)
```

Creation returned `200` with `name`, `pod`, `image`, and `gpus_free` in the
observed response. An accepted creation may initially be `Pending`; poll until
the selected sandbox is `Running`, then verify its hostname and resources
through SSH. Do not launch experiments merely because POST succeeded.
The current schema offers no CPU/RAM PATCH or PUT operation: the tested resize
workflow is an explicitly requested delete and recreation, retaining persistent
home storage and verifying the replacement before use.

### 5.5 Monitor status and lifetime

Select the intended sandbox from `GET /sandboxes`. Observed row fields are:

| Field | Meaning |
|---|---|
| `name`, `pod`, `node` | Logical name, container hostname/pod, and host node |
| `phase` | API-reported state such as `Pending`, `Running`, or `Terminating` |
| `image`, `gpu` | Image reference and allocated GPU count, not GPU utilization |
| `created` | ISO-8601 creation timestamp |
| `expires_at` | Unix seconds, returned as a string in observed responses |
| `last_active` | Unix seconds or `null`; an activity observation, not a job-completion marker |
| `idle_evict_enabled`, `idle_evict_sec` | Idle-reclaim policy; observed threshold `1800` seconds |

CPU/RAM were absent from the observed sandbox rows. Verify per-container
limits through SSH instead of guessing from those rows. API response schemas
are generic JSON objects; handle missing fields and future additions.

A portable read-only poll loop, using the helper above:

```python
import time
from datetime import datetime, timezone

try:
    while True:
        try:
            status, current = api("GET", "/sandboxes")
            print(datetime.now(timezone.utc).astimezone().isoformat(), flush=True)
            show(current["sandboxes"])
        except RuntimeError as exc:
            print(f"Status unavailable: {exc}; retrying in 10s", flush=True)
        time.sleep(10)
except KeyboardInterrupt:
    print("Monitor stopped")
```

For the friendlier dashboard on this workstation, a standalone script exists
at `/mnt/e/school-work-6-y3s2/fyp/lumid_sandbox_monitor.py`, **outside Git**.
It shows sandbox states, expiry countdowns, last activity, GPU allocation,
site GPU availability, and quota. It preserves the last successful data with a
STALE label on failures. It requires Python 3.9+ and no third-party packages.
Other checkouts need their own copy or can use the portable loop above.

The dashboard reads the exported environment, so load the intended credential
explicitly. For this trusted shell-assignment file, run from the repository:

```bash
set -a
. /home/guest1/.config/lumid/credentials.env
set +a
.venv/bin/python3 ../lumid_sandbox_monitor.py --name research-next --interval 10
```

Use `--once` for one snapshot, `--once --json` for complete API JSON with
freshness/error metadata, or `--no-clear` to append terminal updates. Ctrl+C
stops the monitor. Polling is read-only: it does not keep a sandbox active,
extend its TTL, or establish that training finished. An absent sandbox may have
been reclaimed, expired, or deleted. Verify jobs using persistent logs,
completion artifacts, and validators when the sandbox is reachable.

A 24-hour TTL remains subject to idle reclaim; it does not guarantee 24 hours
of idle availability. Convert timestamps with an explicit timezone, for example
`datetime.fromtimestamp(int(row["expires_at"]), timezone.utc).astimezone()`.
Check the local clock before interpreting countdowns.

### 5.6 Delete the exact requested sandbox

Deletion removes compute and can kill jobs. Use it only for the exact sandbox
the owner requested. Inspect the list and, when reachable, its processes and
persistent outputs before deletion. This guide update is not deletion
authorization. Do not replay historical `dev`, `dev2`, or `dev3` deletions.

Using the helper above, for a future explicitly authorized deletion:

```python
name = "research-next"  # Replace with the exact authorized target.
rows = api("GET", "/sandboxes")[1]["sandboxes"]
matches = [row for row in rows if row.get("name") == name]
if len(matches) != 1:
    raise RuntimeError("Target absent or ambiguous; no DELETE sent")
show(matches[0])
status, deleted = api("DELETE", "/sandboxes/" + parse.quote(name, safe=""))
print("HTTP status:", status)
show(deleted)
show(api("GET", "/sandboxes")[1]["sandboxes"])
```

There is **no DELETE request body**. An observed successful response was:

```json
{"deleted":"dev3","home_kept":"home-personai-korolev-tes"}
```

`home_kept` confirms that the named persistent home volume was retained; it
does not preserve running processes or container-local `/root`/`/workspace`
state. A `200` acknowledges the deletion request, while actual cleanup may
still show `Terminating`. Poll GET and quota for disappearance/resource release;
do not repeatedly DELETE just because the row remains. The observed `dev3`
row disappeared after a short delay; `dev` remained Terminating across multiple
checks despite `200`. Report prolonged cleanup with the name, pod, node, and
timestamps; it is not evidence of a completed job.

After a timeout, inspect GET before retrying a mutation. If cleanup still
occupies the pod quota, wait for release before creating a replacement. Never
delete a different sandbox to free quota without the owner's instruction.

### 5.7 Common API failures

| Observation | Next step |
|---|---|
| `401` / `403` | Verify the credential source and token permissions; reads can succeed with a token that cannot mutate |
| `400` resource/scheduling error | Read the returned detail; compare CPU/RAM/GPU requests with eligible node capacity and quota |
| `Pending` after POST | Poll the exact name; creation acceptance is separate from runtime readiness |
| `200` DELETE but `Terminating` remains | Poll GET/quota; report prolonged cleanup instead of blindly retrying DELETE |
| Timeout or connection failure during mutation | Treat outcome as uncertain and inspect the list before retrying |
| API works but SSH returns `502` | Status API and container access are separate paths; do not infer successful experiment completion |

### 5.8 SSH discovery and targeting

List sandboxes:

```bash
ssh -p 31223 gw@lum.id 'sbx ls'
```

Expected format resembles:

```text
NAME                          STATUS   NODE         GPU
sbx-example                   Running  some-node    1
```

The documented interactive command is:

```bash
sbx enter <sandbox-name>
```

or through SSH:

```bash
ssh -tt -p 31223 gw@lum.id 'sbx enter <sandbox-name>'
```

If this interactive path hangs, use one command per SSH connection as a fallback.

Example:

```bash
ssh -p 31223 gw@lum.id 'python3 /workspace/run_backtest.py'
```

This is slower for exploratory work but adequate for batch experiments.

## 6. Filesystem model

Always inspect the environment instead of assuming VM-like persistence:

```bash
ssh -p 31223 gw@lum.id 'pwd; echo "$HOME"; ls -la /home'
```

Operational rule:

```text
/home/<user> -> persistent user data
/datasets    -> shared read-only datasets
/workspace   -> treat as ephemeral
/root        -> treat as ephemeral
```

For long-lived research:

```bash
ssh -p 31223 gw@lum.id 'mkdir -p /home/<user>/backtests'
```

Prefer:

```text
/home/<user>/repo
/home/<user>/backtests
/home/<user>/checkpoints
/home/<user>/logs
```

over container-local directories.

## 7. Data model

Do not confuse external data services with mounted files.

### FinData / Lumid Data / LQT

These are remote services. Selecting/attaching a data source configures the sandbox to reach it.

Check FinData:

```bash
ssh -p 31223 gw@lum.id 'printenv FINDATA_URL'
```

Important quoting rule:

Wrong from a local shell:

```bash
ssh -p 31223 gw@lum.id "echo $FINDATA_URL"
```

The local shell may expand `$FINDATA_URL` before SSH sends the command.

Correct:

```bash
ssh -p 31223 gw@lum.id 'echo "$FINDATA_URL"'
```

For prediction-market data, use the FinData endpoint documented for that dataset.

### `/datasets`

Check:

```bash
ssh -p 31223 gw@lum.id 'ls -la /datasets'
```

Treat `/datasets` as shared/read-only unless site documentation says otherwise.

## 8. Verify Python/GPU environment

Basic checks:

```bash
ssh -p 31223 gw@lum.id 'python3 --version'
ssh -p 31223 gw@lum.id 'python3 -c "import sys; print(sys.executable); print(sys.prefix)"'
ssh -p 31223 gw@lum.id 'nvidia-smi'
```

Check libraries:

```bash
ssh -p 31223 gw@lum.id \
  'python3 -c "import numpy, torch, requests; print(numpy.__version__); print(torch.__version__); print(requests.__version__)"'
```

Do not assume common packages such as `pandas`, `curl`, `scp`, `ss`, or `tmux` exist in a site-default image.

If root and package management are available, install what is needed, but remember that container-local installs may disappear when the sandbox is recreated.

For reproducible environments, prefer a custom image.

## 9. File transfer when remote `scp` is unavailable

Traditional `scp` can fail if the remote image does not contain the `scp` executable.

Portable upload fallback:

```bash
ssh -p 31223 gw@lum.id \
  'cat > /workspace/file.py' \
  < file.py
```

Verify:

```bash
ssh -p 31223 gw@lum.id 'ls -lh /workspace/file.py'
```

For binary files, use a binary-safe transport rather than shell text transformations. Plain SSH stdin redirection with `cat` is binary-safe when no shell transformation is introduced:

```bash
ssh -p 31223 gw@lum.id 'cat > /workspace/archive.bin' < archive.bin
```

For important persistent files, target `/home/<user>` rather than `/workspace`.

## 10. Run long-lived jobs safely

Do not rely on a foreground SSH process surviving disconnects in a useful state.

Preferred pattern:

```bash
ssh -p 31223 gw@lum.id \
  'nohup python3 /home/<user>/backtests/run_backtest.py \
   >/home/<user>/backtests/run_backtest.log \
   2>&1 </dev/null & echo $!'
```

This explicitly sets:

```text
stdin  -> /dev/null
stdout -> log file
stderr -> log file
```

Check process:

```bash
ssh -p 31223 gw@lum.id 'pgrep -af run_backtest.py'
```

Read logs:

```bash
ssh -p 31223 gw@lum.id \
  'tail -n 50 /home/<user>/backtests/run_backtest.log'
```

Stop gracefully:

```bash
ssh -p 31223 gw@lum.id \
  'pkill -TERM -f "[r]un_backtest.py"'
```

Use `SIGKILL` only when graceful termination fails.

## 11. Public ports

When creating a sandbox, specify the **container ports** your application will listen on.

Example site mapping:

```text
lum.id:31501 -> sandbox:8888
lum.id:31502 -> sandbox:6006
```

Your application must listen on:

```text
0.0.0.0:<container-port>
```

not only:

```text
127.0.0.1:<container-port>
```

Example:

```python
server = ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
```

Test internally first.

If `curl` is unavailable, use Python:

```bash
ssh -p 31223 gw@lum.id \
  'python3 -c "import requests; print(requests.get(\"http://127.0.0.1:8888/health\").text)"'
```

Then test externally from the local machine:

```bash
curl http://lum.id:31501/health
curl http://lum.id:31502/health
```

or on Windows:

```powershell
Invoke-WebRequest http://lum.id:31501/health
```

Interpretation:

```text
internal fails, public fails -> application/listener problem
internal works, public TCP fails -> public forwarding/network problem
internal works, public works -> mapping healthy
```

## 12. Reclaim versus delete

Treat these separately.

### Reclaim

Automatic resource recovery when the sandbox is considered idle.

Site policy observed for the `home` GPU sandbox:

- unused GPU sandboxes are reclaimed after about 30 minutes idle;
- an open shell session counts as in use;
- a `tmux` or `screen` session counts as in use, even detached;
- GPU activity counts as in use;
- CPU activity counts as in use;
- a sandbox whose usage cannot be read is not reclaimed.

Do not assume that merely having a sleeping/background process counts as activity. A web server waiting in `accept()` may use effectively zero CPU.

### Delete

Explicitly destroy/remove the sandbox.

Operationally, treat either reclaim or deletion as capable of killing running processes and losing ephemeral container state.

Persistent user data should be stored separately.

### Delete a named sandbox through the PAT-authenticated API

See [Section 5.6](#56-delete-the-exact-requested-sandbox) for the canonical
credential-file client, exact-target DELETE example, and asynchronous cleanup
checks. Use `DELETE https://lum.id/sbx/api/sandboxes/<name>` with Bearer PAT
authentication and no body, only when that named deletion is authorized.

Retained historical owner-supplied response (do not repeat this deletion):

```json
{
  "deleted": "dev2",
  "home_kept": "home-personai-korolev-tes"
}
```

Record the actual status/body and subsequent list state. `home_kept` identifies
retained persistent storage; it does not mean processes survive. A `200` alone
does not prove that the terminating row and occupied quota have disappeared.

### Hard TTL

A hard sandbox lifetime/expiration, if configured, is distinct from idle reclaim. Activity may prevent idle reclaim but should not be assumed to override a hard TTL.

## 13. Reclaim testing

A sleeping HTTP server is not a reliable CPU-activity test.

For a lightweight CPU test, use a low-duty-cycle loop that performs short bursts of work.

For a definitive GPU-activity test, use a short PyTorch matrix-multiplication workload and verify with:

```bash
ssh -p 31223 gw@lum.id 'nvidia-smi'
```

Confirm:

- a Python process appears in the process table;
- GPU memory is allocated;
- GPU utilization rises during work.

Do not leave synthetic keepalive workloads running unnecessarily.

## 14. Known/observed site-default-image behavior

The following were observed on one `home` site-default sandbox image on 2026-09-22. Treat these as image-specific observations, not permanent platform guarantees:

```text
python3   present
numpy     present
torch     present
requests  present
CUDA      working
GPU       visible

pandas    absent
curl      absent
scp       absent
ss        absent
tmux      absent
```

The image contained:

```text
/usr/bin/sh
/usr/bin/bash
/bin/sh -> dash
/bin/bash present
```

Therefore, lack of an interactive session was not simply due to the image having no shell.

## 15. Known interactive-shell issue observed on 2026-09-22

Observed scenario:

- `home` site
- gateway `lum.id:31223`
- running GPU sandbox
- public-key authentication succeeds
- non-interactive commands execute successfully
- `sbx ls` succeeds
- `sbx enter` does not produce a usable interactive shell

Without a TTY:

```bash
ssh -p 31223 gw@lum.id 'sbx enter'
```

the gateway reports:

```text
Unable to use a TTY - input is not a terminal or the right kind of file
```

With a forced TTY:

```bash
ssh -tt -p 31223 gw@lum.id 'sbx enter <sandbox-name>'
```

the command can print the sandbox tip but fail to produce a usable shell.

Workaround:

```bash
ssh -p 31223 gw@lum.id '<one command>'
```

This affects ergonomics but does not necessarily block batch workloads.

Do not generalize this observation to every site/image without reproducing it.

## 16. Backtesting workflow

Recommended workflow:

```text
FinData / remote data
        |
        v
sandbox
  - query/filter data
  - compute factors
  - run walk-forward tests
  - use CPU/GPU
        |
        v
/home/<user>
  - logs
  - metrics
  - checkpoints
  - result files
```

Bootstrap a detached run:

```bash
ssh -p 31223 gw@lum.id \
  'nohup python3 /home/<user>/backtests/run_backtest.py \
   >/home/<user>/backtests/run_backtest.log \
   2>&1 </dev/null & echo $!'
```

Monitor:

```bash
ssh -p 31223 gw@lum.id \
  'tail -n 100 /home/<user>/backtests/run_backtest.log'
```

GPU:

```bash
ssh -p 31223 gw@lum.id 'nvidia-smi'
```

Stop:

```bash
ssh -p 31223 gw@lum.id \
  'pkill -TERM -f "[r]un_backtest.py"'
```

## 17. Troubleshooting checklist

When something fails, isolate layers in this order:

1. **DNS/host**
   ```bash
   ping lum.id
   ```

2. **TCP port**
   ```bash
   nc -vz -w 5 lum.id 31223
   ```

3. **SSH handshake**
   ```bash
   ssh -vvv -o ConnectTimeout=10 -p 31223 gw@lum.id
   ```

4. **Key identity**
   ```bash
   ssh-keygen -lf ~/.ssh/id_ed25519.pub
   ```

5. **Gateway execution**
   ```bash
   ssh -p 31223 gw@lum.id 'sbx ls'
   ```

6. **Sandbox routing**
   ```bash
   ssh -p 31223 gw@lum.id 'hostname'
   ```

7. **Runtime**
   ```bash
   ssh -p 31223 gw@lum.id 'python3 --version'
   ```

8. **GPU**
   ```bash
   ssh -p 31223 gw@lum.id 'nvidia-smi'
   ```

9. **Data**
   ```bash
   ssh -p 31223 gw@lum.id 'printenv FINDATA_URL'
   ```

10. **Internal service**
    ```bash
    ssh -p 31223 gw@lum.id \
      'python3 -c "import requests; print(requests.get(\"http://127.0.0.1:8888/health\").status_code)"'
    ```

11. **Public service**
    ```bash
    curl http://lum.id:<public-port>/health
    ```

Do not jump directly to SSH-key regeneration when the failure is clearly at a later layer.

## 18. Completion checks for an agent

Before declaring a sandbox ready for use, verify:

- correct site and SSH gateway port;
- SSH public key is registered;
- non-interactive SSH command succeeds;
- `hostname` identifies the expected sandbox;
- persistent storage location is known;
- required data endpoint is present;
- required Python packages are available;
- GPU is visible if requested;
- long-running-job logging and shutdown commands are defined;
- public ports are tested internally before external testing;
- reclaim/TTL behavior is understood for unattended jobs;
- important outputs are written to persistent storage.

## References

Primary platform documentation:

- Sandbox API schema: https://lum.id/sbx/openapi.json
- Sandbox guide: https://lum.id/studio/docs/sandboxes
- FlowMesh SSH guide: https://lum.id/studio/docs/flowmesh-ssh

Skill format reference:

- OpenAI skills documentation: https://developers.openai.com/docs/build-skills

<!-- This is a comment line to test the lumid container ssh -->
