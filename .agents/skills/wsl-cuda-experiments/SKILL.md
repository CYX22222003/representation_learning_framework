---
name: wsl-cuda-experiments
description: Select, verify, run, and monitor GPU training experiments in this repository's WSL environment. Use for CUDA training, Python diagnostics, data preparation, baseline sweeps, long-running experiments, or when choosing between WSL and native Windows execution; recommend WSL for repository paths under /mnt/e/.
---

# WSL CUDA Experiments

Prefer WSL for this repository. The project lives under `/mnt/e/`, the virtual environment is `.venv/`, and the repository's Python dependencies and paths are defined for WSL. Do not silently switch to native Windows Python for a training run.

## Scenario Selection

- Use WSL when the current path is `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework` or another Linux-mounted project path.
- Use CUDA when the user requests GPU execution or the experiment is large enough that GPU execution is the intended project workflow.
- Use CPU only when CUDA is unavailable or the user explicitly requests a CPU run. Report that change clearly.
- Use the existing project `.venv`; do not recreate it or install dependencies unless necessary and authorized.

## Verify the Environment

Run these checks before training:

```bash
uname -a
pwd
test -x .venv/bin/python3
.venv/bin/python3 -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
nvidia-smi
```

Use `.venv/bin/python3` for Python and `.venv/bin/pip` for pip. Never use bare `python` or `python3` from the WSL project root. Confirm that PyTorch reports a CUDA build and `torch.cuda.is_available()` is true before passing `--device cuda`.

The normal sandbox may hide GPU access even when WSL and the NVIDIA driver are configured correctly. If `nvidia-smi` or the CUDA Python check fails with an access or operating-system restriction, rerun the diagnostic or training command with the required escalated execution permission. Do not interpret a sandbox denial as proof that CUDA is absent.

## Run Experiments

Start from the project root. Use explicit device, seed, data path, epoch budgets, and run name:

```bash
.venv/bin/python3 src/baselines/mlp_baseline/run_experiment.py \
  --task price \
  --processed-npz data/processed/market_4h_seq64_top50.npz \
  --run-name 2026-08-04-price-sweep \
  --epoch-budgets 15,50,100 \
  --seed 0 \
  --device cuda
```

Use a new run directory for a new configuration. Use `--overwrite` only when the exact target directory has been checked and replacing its artifacts is intended. Preserve checkpoints, histories, predictions, metrics, and summaries.

## Monitor Long Runs

Prefer a foreground tool session with a sufficiently long polling interval. The runner should emit explicit messages when checkpoints are saved, when each checkpoint is evaluated, and when the experiment completes. Do not treat an empty terminal poll as completion or failure; inspect the run directory for `e<budget>/checkpoint.pth`, `metrics.json`, `sweep_metrics.json`, and `summary.md`.

If a long process must be interrupted, identify only the relevant experiment command before terminating it:

```bash
pgrep -af 'src/.*/run_experiment.py'
```

Terminate matching experiment processes only after confirming the target. Do not kill unrelated Python processes or the whole WSL environment. Afterward, report whether artifacts are complete or partial.

## Completion Criteria

Treat a run as complete only when:

1. every requested budget has a checkpoint and history;
2. every budget has predictions and metrics;
3. the root has `sweep_metrics.json` and `summary.md`;
4. the process exits successfully and prints the completion message;
5. diagnostic plots are generated when the experiment plan requires them.

If a run stops after checkpoint creation but before evaluation, preserve it as partial and do not use it as a final result. Re-run with a new name or explicitly overwrite the incomplete directory after confirming its scope.

## Project References

- Runtime rule: `.agents/skills/python-env/SKILL.md`
- Experiment allocation and leakage rules: `.agents/skills/experiment-setup/SKILL.md`
- MLP runner: `src/baselines/mlp_baseline/run_experiment.py`
- MLP plotter: `src/baselines/mlp_baseline/plot_experiment.py`
