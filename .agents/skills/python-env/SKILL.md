---
name: python-env
description: Use when running Python, pip, training, data-preparation, evaluation, or diagnostic commands for this repository, especially when selecting between WSL and Windows execution environments.
---

# Python Environment

Detect the active shell before constructing commands. The project's intended runtime is WSL with a local virtual environment at `.venv/`.

## WSL Rules

When the working directory is under `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework`:

- Invoke Python as `.venv/bin/python3`.
- Invoke pip as `.venv/bin/pip`.
- Do not use bare `python` or `python3`.

Owner-approved xLSTM-Mixer exception: use `.venv-xlstm-mixer/bin/python3`
and `.venv-xlstm-mixer/bin/python3 -m pip` for Phase 6.9 xLSTM scripts/tests.
This isolated environment is intentional; do not mix in `.venv` site-packages.
Its local runner is `scripts_v8/run_phase6_9_xlstm_mixer_experiment.sh`;
default is preparation/admission/manifest-only, and `--execute` trains both
fixed endpoint-only XM-C8 trajectories on original price rows. Legacy XM-MV8
individual scripts remain historical. See `docs/baselines/xLSTM-Mixer/README.md`
and the Phase 6.9 endpoint amendment.

```bash
.venv/bin/python3 scripts/prepare_sequences.py --timeframes 4h --seq-len 64 --top-k 50
.venv/bin/pip install some-package
.venv/bin/python3 -c 'import torch; print(torch.__version__)'
```

## Windows PowerShell

Do not treat `.venv/bin/python3` as a native Windows path. If Codex is running in PowerShell, either execute the project command through WSL or use a verified Windows interpreter only when the user explicitly intends native Windows execution.

Before running a WSL command from PowerShell, verify that WSL is available and that the project path resolves correctly.

## Troubleshooting

If `.venv/bin/python3` is absent inside WSL, report the missing environment before recreating it. Environment recreation and dependency installation may require user approval and network access.
