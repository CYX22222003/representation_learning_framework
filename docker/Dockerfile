# syntax=docker/dockerfile:1

# This image targets non-interactive Lumid Workflow / FlowMesh Python nodes.
# It is intentionally not an SSH-task image.
ARG PYTHON_BASE_IMAGE=python:3.12.3-slim-bookworm
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu22.04
FROM ${PYTHON_BASE_IMAGE} AS framework-runtime

ARG PYTORCH_VERSION=2.12.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu132

LABEL org.opencontainers.image.title="FYP representation learning framework"
LABEL org.opencontainers.image.description="Reproducible runtime for Lumid FlowMesh workflows"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The container dependency set is authoritative for Lumid. PyTorch remains a
# separate install because its wheel index is tied to the target CUDA runtime.
COPY requirements-container.txt /tmp/requirements-container.txt

RUN python -m pip install --no-cache-dir \
        --requirement /tmp/requirements-container.txt

RUN python -m pip install --no-cache-dir \
        "torch==${PYTORCH_VERSION}" \
        --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip check

# Keep the first image focused on code and its runtime. Data, checkpoints, and
# generated experiment artifacts must be supplied explicitly in a later image
# layer or through the workflow's input/artifact mechanism.
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY scripts_v2/ /app/scripts_v2/
COPY scripts_v3/ /app/scripts_v3/
COPY scripts_v4/ /app/scripts_v4/
COPY scripts_v5/ /app/scripts_v5/
COPY scripts_v6/ /app/scripts_v6/
COPY scripts_v8/ /app/scripts_v8/
COPY ablation/ /app/ablation/
COPY tests/ /app/tests/
COPY docs/ /app/docs/

# Bounded, read-only evidence for the FlowMesh audit workflows. These files are
# reports and scalar metrics only: no datasets, predictions, features, or model
# checkpoints are included. Rebuild with a new versioned tag when they change.
COPY experiments/phase5/reports/encoder_pretraining_seed0/summary.json \
    /app/experiments/phase5/reports/encoder_pretraining_seed0/summary.json
COPY experiments/phase6/encoder_variants/reports/complete_seed0/ \
    /app/experiments/phase6/encoder_variants/reports/complete_seed0/
COPY experiments/phase6_5/lstm_capacity/reports/complete_seed0/epoch50_by_walk.csv \
    /app/experiments/phase6_5/lstm_capacity/reports/complete_seed0/epoch50_by_walk.csv
COPY experiments/phase6_5/residual_cnn/reports/complete_seed0/epoch50_by_walk.csv \
    /app/experiments/phase6_5/residual_cnn/reports/complete_seed0/epoch50_by_walk.csv
COPY experiments/phase6_7/downstream/ \
    /app/experiments/phase6_7/downstream/
COPY experiments/phase6_7/reports/frozen_representation_seed0/summary.md \
    /app/experiments/phase6_7/reports/frozen_representation_seed0/summary.md

RUN python -c "import numpy, pandas, pyarrow, pywt, scipy, talib, torch; import src; print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda})"

RUN python -c "from platform_integration.flowmesh_artifact_inspection import AUDITS; print({'flowmesh_audits': sorted(AUDITS)})"

RUN mkdir -p \
        /app/checkpoints \
        /app/data \
        /app/data_new \
        /app/experiments \
        /app/progress_summary \
        /tmp/matplotlib \
        /tmp/.cache \
    && chmod -R a+rX /app

# Do not set USER here. FlowMesh's hardened Python-task bootstrap starts with
# the image default user, prepares its mounted output directory, and then drops
# privileges before calling the workflow entrypoint.

CMD ["python3"]

# Optional local compatibility target. Retain the Python/PyTorch dependency
# set above and add the compiler toolchain required by xlstm's JIT extension.
# CUDA 13.2.1 is a concrete available toolkit patch; the sandbox's nvcc patch
# must still be recorded and compared when that sandbox is created.
FROM ${CUDA_DEVEL_IMAGE} AS cuda-toolchain

FROM framework-runtime AS xlstm-compat
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
        ninja-build \
        libtinfo6 \
        libxml2 \
    && rm -rf /var/lib/apt/lists/*

# Use the Ubuntu 22.04 toolkit donor (older glibc than bookworm). Copying
# Ubuntu 24.04 compiler binaries into bookworm would require a newer glibc.
COPY --from=cuda-toolchain /usr/local/cuda-13.2/ /usr/local/cuda/
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MAX_JOBS=2 \
    TORCH_EXTENSIONS_DIR=/tmp/torch-extensions

RUN python -m pip install --no-cache-dir \
        --requirement /app/src/baselines/xlstm_mixer/requirements.txt \
    && python -m pip check \
    && nvcc --version \
    && c++ --version \
    && ninja --version

# A normal Docker build has no GPU. Compile/run the CUDA backend explicitly
# at container runtime with --gpus all; CPU checks are valid at build time.
RUN python scripts_v8/check_phase6_9_xlstm_mixer_environment.py --device cpu
CMD ["python", "scripts_v8/check_phase6_9_xlstm_mixer_environment.py", "--device", "cpu"]

# Preserve the established default FlowMesh image. The optional CUDA donor
# and xLSTM dependency are only built with --target xlstm-compat.
FROM framework-runtime AS flowmesh
