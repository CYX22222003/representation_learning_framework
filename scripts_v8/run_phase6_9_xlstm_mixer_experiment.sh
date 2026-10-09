#!/usr/bin/env bash
# Active two-walk endpoint-only XM-C8 runner; no training without --execute.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INTERPRETER="${PROJECT_ROOT}/.venv-xlstm-mixer/bin/python3"
DEVICE=cuda
BACKEND=vanilla
MAX_MEMORY_GIB=6
MAX_SECONDS=600
LOG_PATH=""
EXECUTE=0
usage() {
  echo "Usage: bash scripts_v8/run_phase6_9_xlstm_mixer_experiment.sh [--execute] [options]"
  echo "Default: prepare/replay data, admit the runtime, and freeze the matrix only."
  echo "Active contract: direct close[t+8h], original price rows, endpoint-only MSE."
  echo "  --execute                 Train/resume both fixed 50-epoch trajectories, then replay"
  echo "  --python PATH             Interpreter (default: .venv-xlstm-mixer/bin/python3)"
  echo "  --device DEVICE           CUDA device (default: cuda)"
  echo "  --backend vanilla|cuda    sLSTM implementation (default: vanilla; no nvcc needed)"
  echo "  --max-peak-memory-gib N    Admission ceiling (default: 6 GiB)"
  echo "  --max-smoke-seconds N      Per-walk admission ceiling (default: 600 seconds)"
  echo "  --log-path PATH           Persistent log (default: timestamped under phase6_9)"
}
while (($#)); do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --python|--device|--backend|--max-peak-memory-gib|--max-smoke-seconds|--log-path)
      if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --python) INTERPRETER="$2" ;;
        --device) DEVICE="$2" ;;
        --backend) BACKEND="$2" ;;
        --max-peak-memory-gib) MAX_MEMORY_GIB="$2" ;;
        --max-smoke-seconds) MAX_SECONDS="$2" ;;
        --log-path) LOG_PATH="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
cd "${PROJECT_ROOT}"
[[ -x "${INTERPRETER}" ]] || { echo "Missing interpreter: ${INTERPRETER}" >&2; exit 1; }
[[ "${DEVICE}" == cuda* ]] || { echo "Training admission requires a CUDA device" >&2; exit 2; }
[[ "${BACKEND}" == vanilla || "${BACKEND}" == cuda ]] || { echo "Invalid backend" >&2; exit 2; }
mkdir -p "${PROJECT_ROOT}/experiments/phase6_9/xlstm_mixer_endpoint"
exec 9>"${PROJECT_ROOT}/experiments/phase6_9/xlstm_mixer_endpoint/launcher.lock"
flock -n 9 || { echo "Another XM-C8 pipeline is already active" >&2; exit 1; }
if [[ -z "${LOG_PATH}" ]]; then
  LOG_PATH="${PROJECT_ROOT}/experiments/phase6_9/xlstm_mixer_endpoint/logs/pipeline_$(date -u +%Y%m%dT%H%M%S)_${BASHPID}.log"
fi
mkdir -p "$(dirname -- "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap 'status=$?; echo "Pipeline exit: ${status}; log: ${LOG_PATH}"' EXIT
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
stage() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1"
  shift
  "$@"
}
echo "Runtime: ${INTERPRETER}; device: ${DEVICE}; backend: ${BACKEND}; execute: ${EXECUTE}"
echo "Recipe: seed 0, batch 512, Adam 1e-4, epochs 50, snapshots 5/15/50."
echo "Scope: XM-C8 endpoint-only adaptation; original H0/Raw-LSTM controls audited for reuse."
echo "Legacy XM-MV8 artifacts remain unchanged; this runner cannot resume them."
ENDPOINT_ARGS=(--device "${DEVICE}" --backend "${BACKEND}" --admit-training
  --max-peak-memory-gib "${MAX_MEMORY_GIB}" --max-smoke-seconds "${MAX_SECONDS}")
if ((EXECUTE)); then
  ENDPOINT_ARGS+=(--execute)
fi
ENDPOINT_ROOT="${PROJECT_ROOT}/experiments/phase6_9/xlstm_mixer_endpoint"
if ((EXECUTE)) && \
  [[ -f "${ENDPOINT_ROOT}/downstream/absolute_price_h8/walk1/xm_c8/seed0/training_complete.json" ]] && \
  [[ -f "${ENDPOINT_ROOT}/downstream/absolute_price_h8/walk2/xm_c8/seed0/training_complete.json" ]]; then
  stage "Independently replay completed runs and regenerate the matched report" \
    "${INTERPRETER}" scripts_v8/report_phase6_9_xlstm_endpoint.py \
    --device "${DEVICE}" --backend "${BACKEND}"
  exit 0
fi
if stage "Audit original rows/controls, freeze XM-C8, and admit selected backend" \
  "${INTERPRETER}" scripts_v8/run_phase6_9_xlstm_endpoint.py "${ENDPOINT_ARGS[@]}"; then
  :
else
  RUNNER_STATUS=$?
  # Preserve the admitted training fingerprint. Only the known final writer
  # schema error may reach recovery; the standalone report must then prove
  # both completion markers, runtime/data/code/control identities, and replay.
  if ((EXECUTE)) && rg --fixed-strings --line-regexp --quiet \
    "XM-C8 gate failed: 'spearman'" "${LOG_PATH}"; then
    echo "Recovering known final-report schema mismatch; no training guard is waived."
  else
    exit "${RUNNER_STATUS}"
  fi
fi
if ((EXECUTE)); then
  stage "Independently replay all snapshots and write the native-schema comparison" \
    "${INTERPRETER}" scripts_v8/report_phase6_9_xlstm_endpoint.py \
    --device "${DEVICE}" --backend "${BACKEND}"
fi
if ((!EXECUTE)); then
  echo "Readiness pipeline finished. No trajectories launched. Add --execute to train."
fi
