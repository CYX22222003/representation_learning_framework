---
name: lumid-sandbox-experiment-guide
description: Prepare, synchronize, launch, resume, validate, and monitor durable machine-learning experiments on Lumid sandboxes. Use when moving project code or data to Lumid, verifying its Python/CUDA environment, troubleshooting WSL/VPN connectivity to Lumid, designing persistent logs and checkpoints, classifying validation failures, launching detached jobs, diagnosing apparently idle or stopped runs, or arranging bounded or long-running monitoring.
---

# Lumid Sandbox Experiment Guide

Use this skill as the experiment-operations SOP for Lumid. It complements
`lumid_container_guide.md`: that document is the platform reference; this skill
defines the reproducible project workflow.

The required order is:

1. identify the sandbox and persistent storage;
2. prepare and validate the environment;
3. close, transfer, and validate the data dependency set;
4. synchronize committed code through the remote repository;
5. test the complete pipeline without starting the full experiment;
6. launch with persistent logs and resumable checkpoints;
7. monitor the first few intervals and every stage transition;
8. optionally arrange a long-running monitor;
9. validate completion and preserve the evidence.

Do not skip a gate because a previous experiment used the same sandbox.

## Read Before Acting

Read these sources before issuing experiment commands:

- `lumid_container_guide.md` for the current gateway, storage, reclaim, and
  connectivity rules;
- the active experiment plan and relevant implementation/integration plan;
- the launcher, its validators, and the code that resolves input/output paths;
- repository `AGENTS.md` instructions and any task-specific experiment skill.

Read `references/command-templates.md` when concrete SSH, Git, transfer,
launch, monitoring, or verification commands are needed.

## Gate 0: WSL and VPN Connectivity

The native hostel network does not support direct SSH access to the Lumid
sandbox, so the Windows VPN is required for the Lumid connection. Native WSL2
networking may not automatically route through that VPN. Before investigating
Lumid SSH configuration or Codex authentication, compare HTTPS connectivity
from WSL and Windows:

```bash
curl -4 -I --connect-timeout 10 https://chatgpt.com
curl.exe -4 -I --connect-timeout 10 https://chatgpt.com
```

If Linux `curl` times out while `curl.exe` returns an HTTP response, the
Windows/VPN path works but WSL networking does not. This can break Lumid SSH,
cause `workspace routing discovery timed out`, and make `codex doctor` report
unreachable provider endpoints or WebSocket/HTTP timeouts. Do not respond by
reinstalling software or changing SSH or Codex configuration while native WSL
connectivity is still broken.

This machine uses `wsl-vpnkit` for WSL-through-VPN routing. In PowerShell,
inspect its state and start it when stopped:

```powershell
wsl -l -v
wsl -d wsl-vpnkit --cd /app wsl-vpnkit
```

Keep `wsl-vpnkit` running while WSL uses the VPN. Then retest from WSL:

```bash
curl -4 -I --connect-timeout 10 https://chatgpt.com
nc -vz lum.id 31223
```

Any HTTP response, including a Cloudflare `403`, confirms working HTTPS
connectivity; a timeout does not. When connectivity suddenly fails, debug in
this order: confirm the Windows VPN, compare Linux `curl` with Windows
`curl.exe`, check `wsl-vpnkit` with `wsl -l -v`, start it if needed, retest
native WSL HTTPS, retest `lum.id:31223`, and only then investigate SSH or
Codex configuration.

## Establish an Execution Record

Record these values in the working notes or persistent launch log before any
mutation:

- Lumid site and gateway port;
- sandbox name and observed hostname;
- persistent repository path;
- intended Git branch and exact commit SHA;
- virtual-environment path and Python executable;
- experiment launcher, configuration, seed, device, and run identifier;
- all required input roots and expected outputs;
- log path, PID-file path, checkpoint root, and completion markers;
- known reclaim and hard-TTL constraints.

Never rely on “the current sandbox” or “the latest branch” as an identifier.
The gateway may route ordinary commands to the newest sandbox, especially when
more than one sandbox exists. Run `sbx ls` and verify `hostname` before every
state-changing remote action.

## Gate 1: Environment Preparation and Validation

Use `/home/<user>/...` for the repository, virtual environment, copied data,
logs, checkpoints, and results. Treat `/workspace`, `/root`, processes, and
container-local package installs as disposable unless the platform explicitly
guarantees persistence. `/datasets` is shared and read-only.

Verify all of the following:

- the expected sandbox is running and `hostname` matches;
- the repository and `.venv/bin/python3` exist under persistent storage;
- the Python version and required imports match the frozen dependency contract;
- PyTorch reports CUDA availability and the expected GPU when CUDA is required;
- `nvidia-smi` works and no conflicting workload consumes required capacity;
- the persistent volume has enough free space for inputs, caches, checkpoints,
  temporary files, logs, and final artifacts with reasonable headroom;
- the sandbox reclaim and hard-TTL policies are understood;
- the launcher writes every important artifact beneath persistent storage.

Use the existing virtual environment when it is valid. Do not recreate it or
upgrade packages casually. Record installed versions for dependencies whose
numerical behavior affects the method.

Environment readiness is a gate, not a training run. Use import probes,
focused unit tests, manifest-only commands, and small smoke inputs only.

## Gate 2: Data Preparation, Synchronization, and Validation

Build a dependency closure from code, not memory. Trace every path read by the
launcher and validators, including:

- source arrays or processed datasets;
- labels and row-identity metadata;
- feature stores;
- train-only scalers or transforms;
- manifests, hashes, provenance, and split definitions;
- immutable comparator checkpoints, predictions, or metrics;
- caches required to resume rather than rebuild;
- configuration files and lookup tables.

Create an inventory containing repository-relative path, type, byte size, and
SHA-256 hash. Compare the inventory with the sandbox before transferring.

Transfer exact required files to the same repository-relative locations. Do
not blindly copy a broad parent directory: hidden work directories, temporary
caches, and incomplete artifacts can waste bandwidth or shadow canonical
files. Create destination parents explicitly.

For unreliable networks:

- preflight SSH and confirm routing immediately before transfer;
- prefer a few controlled sequential streams over many parallel streams;
- use resumable transfer support such as `rsync --partial --append-verify` when
  available at both ends;
- otherwise transfer exact files and assume an interrupted `scp` is incomplete;
- never accept existence alone as success—zero-byte and partially sized files
  are invalid;
- retry only the missing or hash-mismatched files;
- avoid starting a second transfer over a destination still being written.

After transfer, compare byte sizes and SHA-256 hashes at both ends, then run the
project’s standalone data/feature validators. A completed transfer is not the
same as a valid experiment input.

If data is suitable for a shared corpus, publish it through the documented NAS
workflow and consume it read-only from `/datasets`. For an ad hoc project
experiment, a repository-local Git-ignored path under persistent `/home` is
acceptable, but it must still be inventoried and validated.

## Gate 3: Code and Implementation Synchronization

Use the remote Git repository as the source of truth for tracked code:

For this project, the sandbox GitHub identity is persistent at
`/home/personai-korolev-tes/.ssh/id_ed25519`. The sandbox may report
`$HOME=/root`, so Git must not rely on the default identity or host-key paths.
Keep `known_hosts` beside the persistent key and configure the persistent
clone with:

```bash
cd /home/personai-korolev-tes/representation_learning_framework
git remote set-url origin \
  git@github.com:CYX22222003/representation_learning_framework.git
git config core.sshCommand \
  'ssh -i /home/personai-korolev-tes/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/personai-korolev-tes/.ssh/known_hosts -o StrictHostKeyChecking=yes'
```

Before the first Git mutation in a recreated sandbox, confirm permissions,
fingerprint the public key, run `ssh -T` with the explicit identity and
persistent host file, and verify `git ls-remote origin`. GitHub's successful
`ssh -T` response normally exits with status `1`. Never display or transfer
the private key.

1. On the local machine, inspect the branch and worktree.
2. Test the intended changes.
3. Commit them and push the exact branch to `origin`.
4. In the sandbox, verify the sandbox identity, repository path, branch, and
   worktree before updating.
5. Fast-forward with `git pull --ff-only origin <branch>`.
6. Confirm local, origin, and sandbox resolve to the same commit SHA.

Do not synchronize tracked implementation code with ad hoc `scp`, edit the
sandbox copy as an unrecorded hotfix, reset away remote artifacts, or pull over
unknown tracked changes. Git-ignored experiment artifacts may coexist with a
clean tracked worktree, but inspect them before assuming paths are reusable.

If the code changes after data transfer, rerun dependency closure: a new commit
may change required files, schemas, manifests, or output locations.

## Gate 4: Test and Verify the Experimental Pipeline

Exercise the same entry points the full run will use without launching the
full experiment:

- focused model and pipeline tests;
- manifest-only/bootstrap or dry-run mode;
- a bounded CPU smoke test where appropriate;
- a minimal CUDA forward/backward probe on the target GPU;
- cache and source-provenance validation;
- input identity, shape, dtype, finiteness, and hash checks;
- checkpoint save/load and resume smoke tests;
- validation of required immutable comparator artifacts;
- verification that every stage hands its declared outputs to the next stage.

Do not substitute a component test for a pipeline test. The final preflight
must resolve real production paths and configuration while remaining bounded.

Freeze resource limits and numerical replay tolerances before seeing final
downstream results. Save the preflight or admission manifest persistently.

## Build for Persistent Logging and Resume

A long experiment launcher must be restartable and explicit about state.

Persist at least:

- a timestamped combined stdout/stderr log;
- the exact launch command, commit SHA, configuration, environment versions,
  device information, and data/cache hashes;
- a PID file or launch record;
- per-stage start, success, failure, skip, and elapsed-time events;
- training history and finite-loss diagnostics;
- frequent resume checkpoints plus all precommitted evaluation checkpoints;
- validator results, warnings, metrics, predictions, and completion markers.

Save a resume checkpoint at least once per epoch for costly training unless a
documented storage/performance constraint justifies another interval. Include
model, optimizer, scheduler, scaler, epoch/step, RNG and sampler state, frozen
configuration, and input provenance. Write to a temporary file and atomically
replace the canonical checkpoint. Write `training_complete` only after all
required artifacts are durably stored.

Make reruns idempotent: validate and reuse complete stages, resume valid
partial stages, and reject incompatible artifacts. Never infer completeness
from a directory merely existing.

## Validation Severity: Warn Without Hiding Critical Failure

Do not make all validation nonfatal. Classify every validation result.

Fatal conditions must stop the affected pipeline before invalid results are
used:

- missing, truncated, corrupt, or internally inconsistent artifacts;
- source, row-identity, split, configuration, dependency, or provenance drift;
- wrong shape/dtype, non-finite data, loss, gradient, feature, or prediction;
- material replay disagreement beyond precommitted tolerances;
- failed checkpoint integrity or incompatible resume state;
- OOM, CUDA/runtime failure, or unmet hardware/resource admission;
- an absent required downstream dependency or immutable comparator.

Warnings may continue only for pre-approved, scientifically benign conditions,
for example small CPU/CUDA numerical drift inside frozen scale-aware relative
L2 and cosine bounds, or a regenerated feasibility-manifest file hash whose
stable source, cache, dependency, batch, limits, and admission semantics all
revalidate.

For warnings:

- emit a visible warning in the persistent log;
- write structured evidence to a validation JSON artifact;
- record observed values, thresholds, severity, rationale, and continuation;
- keep the standalone validator’s final status unambiguous;
- never relax thresholds after inspecting results.

A launcher should continue after recorded warnings and stop on fatal failures.
Do not append `|| true` to broad validation or training commands; handle
severity inside the validator.

## Launch and First-Few-Shots Monitoring

Launch from the persistent repository with stdin detached and stdout/stderr
redirected to a persistent timestamped log. Record the launcher PID. `nohup`
survives an SSH disconnect; it does not survive sandbox deletion, reclaim,
host failure, or a hard TTL.

Monitor closely for the first few intervals and each stage transition. Confirm:

- the detached launcher and expected child process exist;
- the log file grows and shows the intended commit/configuration;
- batches or epochs advance and losses remain finite;
- checkpoints appear at the intended cadence and can be loaded;
- GPU utilization, memory, temperature, and power are plausible for the stage;
- CPU, RAM, disk, and I/O do not approach unsafe limits;
- validation warnings are recorded without terminating the pipeline;
- the next stage starts only after its required completion marker.

An idle GPU is not automatically a failure: preprocessing, CWT/FFT/cache
generation, validation, feature writing, and downstream preparation may be
CPU- or I/O-bound. Correlate process state, CPU use, log freshness, files, and
the declared stage.

If a process disappears, inspect the log, exit/failure marker, most recent
checkpoint, and output integrity before deciding it completed. Also reconfirm
the sandbox hostname; gateway routing to a different sandbox can make a live
process appear absent.

## Optional Long-Running Monitor

Arrange recurring monitoring only when the user asks for it. A monitor should
poll at a sensible interval and report or alert on:

- launcher/worker exit;
- fatal text or explicit failure markers;
- stale logs or checkpoints beyond a stage-appropriate threshold;
- disk exhaustion, OOM evidence, or abnormal GPU state;
- sandbox TTL approaching;
- successful stage or experiment completion.

The monitor must write its own log under persistent storage. It must not restart
training, change configuration, relax validation, or create synthetic CPU/GPU
keepalive load unless the user explicitly authorizes that behavior. A sleeping
monitor is not proof of workload activity and may not prevent reclaim.

## Resume and Recovery

On failure:

1. verify the sandbox identity and current commit;
2. inspect the persistent log and structured failure artifact;
3. verify the last checkpoint and every completed stage;
4. fix only the root cause, such as a missing dependency or incomplete file;
5. rerun preflight for anything affected by the fix;
6. invoke the same resumable launcher and confirm completed stages are reused;
7. monitor the resumed stage through at least one new checkpoint.

Preserve partial artifacts until their diagnostic or recovery value is known.
Do not delete or overwrite them merely to make a rerun start cleanly.

## Completion Criteria

Declare the experiment complete only when:

- the launcher exited successfully and wrote its completion marker;
- every requested stage and checkpoint exists;
- final predictions, metrics, and reports exist where required;
- standalone replay and provenance validation pass or contain only approved,
  structured warnings;
- artifact hashes and sizes are recorded;
- no required downstream stage remains unexecuted;
- the final commit, configuration, data identity, resource record, and log path
  are reported to the user.

If the sandbox expires after durable checkpoints were saved, report the run as
interrupted and resumable—not complete and not lost.

## Relationship to Other Skills

- Use `experiment-setup` for leakage, allocation, and evaluation-order rules.
- Use `baseline-comparison` for fair comparator and claim design.
- Use `python-env` for repository Python-environment conventions.
- Use `wsl-cuda-experiments` only for local WSL execution, not as a substitute
  for this Lumid SOP.
- Use `sync-docs` when experiment outcomes or approved operational decisions
  change project documentation.
