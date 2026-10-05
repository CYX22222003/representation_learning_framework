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

- Sandbox guide: https://lum.id/studio/docs/sandboxes
- FlowMesh SSH guide: https://lum.id/studio/docs/flowmesh-ssh

Skill format reference:

- OpenAI skills documentation: https://developers.openai.com/docs/build-skills

<!-- This is a comment line to test the lumid container ssh -->
