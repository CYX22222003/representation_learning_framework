#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

.venv/bin/python3 scripts_v8/prepare_phase6_9_sgn_initialization.py
.venv/bin/python3 scripts_v8/validate_phase6_9_sgn_initialization.py
.venv/bin/python3 scripts_v8/bootstrap_phase6_9_sgn.py --device cpu

if [[ "${1:-}" == "--admit-training" || "${1:-}" == "--execute" ]]; then
  .venv/bin/python3 scripts_v8/bootstrap_phase6_9_sgn.py --device cuda --admit-training
fi

if [[ "${1:-}" == "--execute" ]]; then
  .venv/bin/python3 scripts_v8/execute_phase6_9_sgn.py --device cuda
  .venv/bin/python3 scripts_v8/validate_phase6_9_sgn_same_backend.py --device cuda
  .venv/bin/python3 scripts_v8/report_phase6_9_sgn.py
  .venv/bin/python3 scripts_v8/report_phase6_9_sgn_matched.py
fi
