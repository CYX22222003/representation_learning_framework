#!/usr/bin/env bash
# Run the complete TimeDART-only Phase 6.7 extension in dependency order.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python3"

FEATURE_BATCH_SIZE=1024
MAX_PEAK_MEMORY_GIB=23
MAX_SMOKE_SECONDS=600
LOG_PATH=""

usage() {
  echo "Usage: bash scripts_v6/run_phase6_7_timedart_experiment.sh [options]"
  echo
  echo "Options:"
  echo "  --feature-batch-size N     Frozen inference batch size (default: 1024)"
  echo "  --max-peak-memory-gib N    CUDA admission ceiling (default: 23)"
  echo "  --max-smoke-seconds N      Per-walk CUDA smoke ceiling (default: 600)"
  echo "  --log-path PATH            Persistent log path (default: timestamped)"
  echo "  -h, --help                 Show this message"
}

while (($#)); do
  case "$1" in
    --feature-batch-size)
      FEATURE_BATCH_SIZE="$2"
      shift 2
      ;;
    --max-peak-memory-gib)
      MAX_PEAK_MEMORY_GIB="$2"
      shift 2
      ;;
    --max-smoke-seconds)
      MAX_SMOKE_SECONDS="$2"
      shift 2
      ;;
    --log-path)
      LOG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing executable project interpreter: ${PYTHON}" >&2
  exit 1
fi
if [[ ! "${FEATURE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Feature batch size must be a positive integer" >&2
  exit 2
fi

if [[ -z "${LOG_PATH}" ]]; then
  LOG_PATH="${PROJECT_ROOT}/experiments/phase6_7/logs/timedart_full_pipeline_$(date -u +%Y%m%dT%H%M%SZ).log"
elif [[ "${LOG_PATH}" != /* ]]; then
  LOG_PATH="${PROJECT_ROOT}/${LOG_PATH}"
fi
mkdir -p "$(dirname -- "${LOG_PATH}")"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

on_exit() {
  status=$?
  if ((status == 0)); then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TimeDART pipeline completed."
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TimeDART pipeline stopped on a functional stage failure (${status})." >&2
    echo "Advisory validation findings do not stop this launcher; inspect the failing preparation/training/extraction stage." >&2
  fi
  echo "Log: ${LOG_PATH}"
}
trap on_exit EXIT

stage() {
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] === $1 ==="
  shift
  "$@"
}

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

echo "Project root: ${PROJECT_ROOT}"
echo "Log: ${LOG_PATH}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
echo "Feature batch size: ${FEATURE_BATCH_SIZE}"
echo "Admission limits: ${MAX_PEAK_MEMORY_GIB} GiB, ${MAX_SMOKE_SECONDS}s per walk"
echo "Scope: TimeDART inputs and artifacts only; no H0, SaURL, or LWA model artifact is loaded."

stage "CUDA preflight" \
  "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))'

stage "Freeze the six-file TimeDART data dependency closure" \
  "${PYTHON}" scripts_v6/prepare_phase6_7_timedart_data.py

stage "CPU model and pipeline smoke" \
  "${PYTHON}" scripts_v6/audit_phase6_7_timedart.py --device cpu

stage "Fixed batch-16 CUDA admission smoke" \
  "${PYTHON}" scripts_v6/audit_phase6_7_timedart.py \
  --device cuda \
  --admit-training \
  --max-peak-memory-gib "${MAX_PEAK_MEMORY_GIB}" \
  --max-smoke-seconds "${MAX_SMOKE_SECONDS}"

stage "Freeze the two-run TimeDART encoder matrix" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_timedart.py --device cpu

stage "Train or resume both TimeDART encoder trajectories" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_timedart.py --device cuda --execute

stage "Advisory encoder replay (warnings do not block)" \
  "${PYTHON}" scripts_v6/validate_phase6_7_timedart.py

stage "Build or reuse both 170-wide TimeDART master stores" \
  "${PYTHON}" scripts_v6/prepare_phase6_7_timedart_features.py \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE}"

stage "Advisory feature replay (warnings do not block)" \
  "${PYTHON}" scripts_v6/validate_phase6_7_timedart_features.py \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE}" \
  --replay

stage "Freeze the six-run TimeDART downstream matrix" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_timedart_downstream.py --device cpu

stage "Train or reuse all six TimeDART downstream trajectories" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_timedart_downstream.py --device cuda --execute

stage "Advisory downstream replay (warnings do not block)" \
  "${PYTHON}" scripts_v6/validate_phase6_7_timedart_downstream.py

echo
echo "Artifacts: ${PROJECT_ROOT}/experiments/phase6_7"

