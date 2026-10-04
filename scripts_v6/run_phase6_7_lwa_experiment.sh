#!/usr/bin/env bash
# Run the complete approved Phase 6.7 LWA-Frozen experiment in dependency order.
#
# This script is resumable through the Python entry points: completed caches,
# pretraining trajectories, stores, and downstream runs are validated and
# reused. All artifacts and this script's log stay beneath the repository's
# experiments/phase6_7 directory. In Lumid the repository is under persistent
# /home storage.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python3"

CACHE_CHUNK_SIZE=256
FEATURE_BATCH_SIZE=1024
MAX_PEAK_MEMORY_GIB=23
MAX_SMOKE_SECONDS=600
LOG_PATH=""

usage() {
  echo "Usage: bash scripts_v6/run_phase6_7_lwa_experiment.sh [options]"
  echo
  echo "Options:"
  echo "  --cache-chunk-size N       CWT cache rows per CPU chunk (default: 256)"
  echo "  --feature-batch-size N     Frozen inference batch size (default: 1024)"
  echo "  --max-peak-memory-gib N    CUDA admission ceiling (default: 23)"
  echo "  --max-smoke-seconds N      Per-walk CUDA smoke ceiling (default: 600)"
  echo "  --log-path PATH            Persistent log path (default: timestamped)"
  echo "  -h, --help                 Show this message"
}

while (($#)); do
  case "$1" in
    --cache-chunk-size)
      CACHE_CHUNK_SIZE="$2"
      shift 2
      ;;
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

for value in "${CACHE_CHUNK_SIZE}" "${FEATURE_BATCH_SIZE}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch and chunk sizes must be positive integers: ${value}" >&2
    exit 2
  fi
done

if [[ -z "${LOG_PATH}" ]]; then
  LOG_PATH="${PROJECT_ROOT}/experiments/phase6_7/logs/lwa_full_pipeline_$(date -u +%Y%m%dT%H%M%SZ).log"
elif [[ "${LOG_PATH}" != /* ]]; then
  LOG_PATH="${PROJECT_ROOT}/${LOG_PATH}"
fi
mkdir -p "$(dirname -- "${LOG_PATH}")"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

on_exit() {
  status=$?
  if ((status == 0)); then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] LWA pipeline completed successfully."
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] LWA pipeline failed with status ${status}." >&2
    echo "Resume with the same command after resolving the reported error." >&2
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
echo "Cache chunk size: ${CACHE_CHUNK_SIZE}"
echo "Feature batch size: ${FEATURE_BATCH_SIZE}"
echo "Admission limits: ${MAX_PEAK_MEMORY_GIB} GiB, ${MAX_SMOKE_SECONDS}s per walk"

stage "CUDA and dependency preflight" \
  "${PYTHON}" -c 'import pywt, torch; assert pywt.__version__ == "1.8.0"; assert torch.cuda.is_available(); print(torch.__version__); print(torch.cuda.get_device_name(0))'

stage "Build or replay both walk-specific FFT/CWT caches" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_lwa.py \
  --device cpu \
  --prepare-cache \
  --cache-chunk-size "${CACHE_CHUNK_SIZE}"

stage "Fixed physical-batch-128 CUDA admission smoke" \
  "${PYTHON}" scripts_v6/audit_phase6_7_lwa.py \
  --device cuda \
  --admit-training \
  --max-peak-memory-gib "${MAX_PEAK_MEMORY_GIB}" \
  --max-smoke-seconds "${MAX_SMOKE_SECONDS}"

stage "Train or resume both 50+50 epoch LWA trajectories" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_lwa.py \
  --device cuda \
  --execute

stage "Replay both LWA pretraining trajectories" \
  "${PYTHON}" scripts_v6/validate_phase6_7_lwa.py

stage "Extract both 384-wide three-task LWA master stores" \
  "${PYTHON}" scripts_v6/prepare_phase6_7_lwa_features.py \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE}"

stage "Replay both LWA master stores" \
  "${PYTHON}" scripts_v6/validate_phase6_7_lwa_features.py \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE}"

stage "Freeze the six-run LWA downstream matrix" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_lwa_downstream.py \
  --device cpu

stage "Train or validate all six LWA downstream trajectories" \
  "${PYTHON}" scripts_v6/bootstrap_phase6_7_lwa_downstream.py \
  --device cuda \
  --execute

stage "Replay all LWA downstream checkpoints and predictions" \
  "${PYTHON}" scripts_v6/validate_phase6_7_lwa_downstream.py

echo
echo "Artifacts: ${PROJECT_ROOT}/experiments/phase6_7"
