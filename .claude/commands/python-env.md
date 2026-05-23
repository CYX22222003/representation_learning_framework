This project runs inside WSL (Windows Subsystem for Linux). The default Python environment is a local virtual environment at `.venv/` in the project root.

**Rules to follow for every session:**

1. Never use bare `python` or `python3` — these are not on PATH and will exit 127.
2. Always invoke Python as `.venv/bin/python3` and pip as `.venv/bin/pip`.
3. The project root is `/mnt/e/School-Work-6-Y3S2/FYP/representation_learning_framework` inside WSL.

**Examples:**
```bash
.venv/bin/python3 scripts/prepare_sequences.py --timeframes 4h --seq-len 64 --top-k 50
.venv/bin/pip install some-package
.venv/bin/python3 -c "import torch; print(torch.__version__)"
```

**Troubleshooting:**
- If `.venv/bin/python3` is missing, the venv may need to be recreated: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Confirm WSL is the active shell (working directory should begin with `/mnt/e/`)
