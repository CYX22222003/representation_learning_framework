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
git branch --show-current
git status --short
git rev-parse HEAD
git fetch origin <branch>
git pull --ff-only origin <branch>
git rev-parse HEAD
```

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
