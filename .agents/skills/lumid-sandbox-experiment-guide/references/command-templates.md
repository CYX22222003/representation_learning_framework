# Lumid Experiment Command Templates

These are templates, not fixed project values. Replace angle-bracket values and
verify them against `lumid_container_guide.md` and the active experiment plan.

## 1. Identify the target

The current home-site gateway is normally:

```bash
ssh -p 31223 gw@lum.id 'sbx ls'
ssh -p 31223 gw@lum.id 'hostname; whoami; pwd'
```

For an explicitly named interactive sandbox:

```bash
ssh -tt -p 31223 gw@lum.id 'sbx enter <sandbox-name>'
```

Ordinary gateway commands may route to the newest sandbox. If multiple
sandboxes are running, do not mutate state until the returned hostname has
been matched to the intended sandbox.

## 2. Local code synchronization

```bash
git branch --show-current
git status --short
git rev-parse HEAD
<focused-test-command>
git add <intended-paths>
git commit -m '<message>'
git push origin <branch>
git ls-remote origin refs/heads/<branch>
```

Do not stage unrelated user changes.

## 3. Sandbox code synchronization

Run inside the confirmed sandbox:

```bash
cd /home/<user>/<repository>
ssh -i /home/personai-korolev-tes/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/home/personai-korolev-tes/.ssh/known_hosts \
  -o StrictHostKeyChecking=yes \
  -T git@github.com
git remote set-url origin \
  git@github.com:CYX22222003/representation_learning_framework.git
git config core.sshCommand \
  'ssh -i /home/personai-korolev-tes/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/personai-korolev-tes/.ssh/known_hosts -o StrictHostKeyChecking=yes'
git branch --show-current
git status --short
git rev-parse HEAD
git fetch origin <branch>
git pull --ff-only origin <branch>
git rev-parse HEAD
```

Treat GitHub's authenticated `ssh -T` message with exit status `1` as a
successful connectivity probe. Never print or transfer the private key. For a
different Lumid user or repository, replace all project-specific paths and the
GitHub repository URL explicitly.

The final SHA must match the local and origin SHA. Stop if tracked sandbox
changes would be overwritten or require a merge.

## 4. Environment and capacity checks

Run inside the confirmed sandbox repository:

```bash
hostname
pwd
test -x .venv/bin/python3
.venv/bin/python3 --version
.venv/bin/python3 -c 'import sys; print(sys.executable); print(sys.prefix)'
.venv/bin/python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
nvidia-smi
df -h /home/<user>/<repository>
du -sh /home/<user>/<repository>/<input-root> /home/<user>/<repository>/<output-root>
```

Check method-specific packages in one bounded import probe, for example:

```bash
.venv/bin/python3 -c 'import numpy, torch, pywt; print(numpy.__version__, torch.__version__, pywt.__version__)'
```

## 5. Exact data inventory

On the local machine:

```bash
stat -c '%s %n' <file-1> <file-2>
sha256sum <file-1> <file-2> > /tmp/<experiment>-data.sha256
```

On the sandbox after transfer:

```bash
cd /home/<user>/<repository>
stat -c '%s %n' <file-1> <file-2>
sha256sum -c <persistent-inventory-path>
<project-data-validator>
```

The inventory itself should be copied to a persistent experiment-provenance
directory and retained with the run.

## 6. Transfer over a stable connection

Preflight immediately before copying:

```bash
ssh -o ConnectTimeout=15 -p 31223 gw@lum.id 'hostname; df -h /home/<user>'
```

Create exact parents:

```bash
ssh -p 31223 gw@lum.id 'mkdir -p /home/<user>/<repository>/<destination-parent>'
```

Copy exact files:

```bash
scp -P 31223 <local-file> \
  gw@lum.id:/home/<user>/<repository>/<exact-destination>
```

If `rsync` is available at both ends, prefer a resumable exact-file transfer:

```bash
rsync -ah --partial --append-verify --info=progress2 \
  -e 'ssh -p 31223' \
  <local-file> gw@lum.id:/home/<user>/<repository>/<exact-destination>
```

After a timeout, do not infer completion from silence or destination existence.
Compare size and SHA-256. With plain `scp`, overwrite or retry the mismatched
exact file; do not launch against a partial destination.

Avoid multiple large parallel streams on an unstable network. A single stream
is slower but easier to resume, diagnose, and verify.

## 7. Preflight pipeline

Use project-specific entry points:

```bash
.venv/bin/python3 <focused-test-runner>
.venv/bin/python3 <bootstrap-script> --device cpu
.venv/bin/python3 <audit-script> --device cuda <bounded-admission-options>
.venv/bin/python3 <validator-script>
```

Confirm these commands do not start the full experiment unless the explicit
execution flag is supplied.

## 8. Detached launch

Run inside the confirmed sandbox repository:

```bash
RUN_TAG=<experiment>-$(date -u +%Y%m%dT%H%M%SZ)
LOG_DIR=/home/<user>/<repository>/experiments/<phase>/logs
mkdir -p "$LOG_DIR"
nohup bash <launcher-script> \
  --log-path "$LOG_DIR/${RUN_TAG}.log" \
  >"$LOG_DIR/${RUN_TAG}.nohup.log" 2>&1 </dev/null &
RUN_PID=$!
printf '%s\n' "$RUN_PID" >"$LOG_DIR/${RUN_TAG}.pid"
printf 'pid=%s log=%s\n' "$RUN_PID" "$LOG_DIR/${RUN_TAG}.log"
```

If the launcher does not accept `--log-path`, preserve the combined nohup log
and change the launcher in version control so future stage records are explicit.

## 9. First-few-shots monitoring

```bash
ps -o pid,ppid,etime,stat,%cpu,%mem,cmd -p <launcher-pid>
pgrep -a -P <launcher-pid>
tail -n 80 <persistent-log>
stat -c '%y %s %n' <persistent-log> <latest-checkpoint>
nvidia-smi
df -h /home/<user>/<repository>
```

Repeat a few times at stage-appropriate intervals. Keep individual waits below
one minute while actively collaborating with the user.

For CPU/I/O-heavy stages:

```bash
ps -eo pid,ppid,etime,stat,%cpu,%mem,cmd --sort=-%cpu | head -20
```

Do not diagnose failure from low GPU utilization alone.

## 10. Bounded long-running monitor

When explicitly requested, a read-only monitor may periodically collect:

```bash
date -u
pgrep -af '<unique-launcher-pattern>'
tail -n 40 <persistent-log>
stat -c '%y %s %n' <persistent-log> <latest-checkpoint>
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
df -h /home/<user>/<repository>
```

Use the product's recurring-monitoring facility when available. Otherwise keep
the monitor observational and persist its output. It must not auto-restart the
experiment or act as a synthetic keepalive.

## 11. Resume checks

```bash
hostname
cd /home/<user>/<repository>
git rev-parse HEAD
tail -n 120 <persistent-log>
find <experiment-output-root> -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort
<checkpoint-validator>
<data-and-provenance-validator>
```

Rerun the same idempotent launcher only after these checks. Verify the log says
which stages were reused, resumed, or restarted and why.

## 12. Completion verification

```bash
pgrep -af '<unique-launcher-pattern>'
tail -n 120 <persistent-log>
<standalone-replay-validator>
find <experiment-output-root> -type f -name '*complete*' -o -name 'metrics.json' -o -name 'predictions.npz'
```

No matching PID can mean success, failure, reclaim, or misrouting. Completion
requires the final log event, required artifacts, and standalone validation.

## 13. Sandbox lifecycle through the API (no Studio)

The canonical standard-library Python client and executable snippets live in
`lumid_container_guide.md`, Sections 5.1--5.7. Run the shared helper and operation
snippets in the same Python session/script. From the repository, use
`.venv/bin/python3`; no `requests` installation is needed.

Credential file on this workstation:
`/home/guest1/.config/lumid/credentials.env`, key `LUMID_PAT`. Read it explicitly
for the selected write credential; an existing exported `LUMID_PAT` or
`/home/guest1/.lumid_env` can hold another token. Never print credential values.
The observed mutation scope is `sandbox:sandboxes:write` (or `*`).

| Step | Request / verification |
|---|---|
| Identify | `GET https://lum.id/sbx/api/whoami` (identity, not token scopes) |
| Inspect capacity and names | `GET https://lum.id/sbx/api/sandboxes` |
| Inspect resource quota | `GET https://lum.id/sbx/api/quota` |
| Create when requested | `POST https://lum.id/sbx/api/sandboxes` with JSON below |
| Poll readiness | GET the list and select the exact `name`; wait for `Running` |
| Verify runtime | SSH hostname, cgroup limits, GPU, persistent home |
| Delete when requested | `DELETE https://lum.id/sbx/api/sandboxes/<name>`; no body |
| Verify cleanup | GET list/quota; a `200` may precede disappearance |

Example creation body for an authorized future sandbox:

```json
{
  "name": "research-next",
  "image": "pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime",
  "cpu": 8,
  "memory_gi": 24,
  "gpu": 1,
  "ttl_hours": 24,
  "data_sources": [],
  "ports": []
}
```

Inspect the live image/capacity catalog first. Schema maxima differ from actual
GPU-node capacity: the observed site rejected 32 GiB (29 GiB maximum eligible
machine); 8 CPUs / 24 GiB / one GPU succeeded on 2026-10-18. Choose resources
for the task rather than relying on defaults (2 CPUs / 8 GiB / no GPU /
8-hour TTL). A 24-hour TTL remains subject to idle reclaim.

After API readiness, verify the expected hostname before interpreting these
read-only runtime checks inside the sandbox:

```bash
hostname
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/memory.max
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -ld /home/<user>
```

For cgroup v2, `800000 100000` means an 8-core CPU quota; `25769803776` bytes
means 24 GiB system RAM. Those files may differ on other cgroup layouts; do not
use host-wide `free`/`nproc` alone as proof of container limits.

The local, Git-external terminal monitor is
`/mnt/e/school-work-6-y3s2/fyp/lumid_sandbox_monitor.py`. With the trusted shell
credential file, run from this repository:

```bash
set -a
. /home/guest1/.config/lumid/credentials.env
set +a
.venv/bin/python3 ../lumid_sandbox_monitor.py --name research-next --interval 10
```

Use `--once --json` for a complete JSON snapshot with freshness/error metadata,
or `--no-clear` for appended updates. Other workstations need their own script
copy or the portable poll loop in guide Section 5.5. The monitor reads only
status/quota and marks retained data stale after errors; it neither extends
TTL nor proves training completion.

Disable HTTP redirects, use bounded timeouts, and record status/body without
credentials. Do not blindly retry an uncertain POST/DELETE; inspect GET first.
DELETE `200` with `home_kept` preserves the reported home volume, while
processes and ephemeral paths remain disposable. Poll for quota release before
recreation. Report prolonged `Terminating` state with name/pod/node/timestamps.
Do not replay historical `dev`, `dev2`, or `dev3` deletions or create test
sandboxes merely to check documentation. A docs update grants no mutation.
