#!/usr/bin/env bash
# Isolated WSL runtime for vanilla xLSTM-Mixer; never launches training.
set -euo pipefail
cd "$(dirname "$0")/.."

xlstm_env=.venv-xlstm-mixer
xlstm_python="$xlstm_env/bin/python3"
if [[ ! -x "$xlstm_python" ]]; then
    if [[ -e "$xlstm_env" ]]; then
        echo "Refusing to replace existing incomplete environment: $xlstm_env" >&2
        exit 1
    fi
    .venv/bin/python3 -m venv "$xlstm_env"
fi
"$xlstm_python" -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
    pip==25.2 setuptools==80.9.0 wheel==0.45.1
"$xlstm_python" -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
    'torch==2.12.1+cu126' --index-url https://download.pytorch.org/whl/cu126 \
    -c scripts_v8/requirements-xlstm-local-lock.txt
"$xlstm_python" -m pip install --disable-pip-version-check --no-cache-dir --no-compile \
    -r scripts_v8/requirements-xlstm-local.txt \
    -c scripts_v8/requirements-xlstm-local-lock.txt

# Resolve this venv's installation path without importing the broken loader.
xlstm_site=$("$xlstm_python" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')
xlstm_loader="$xlstm_site/xlstm/blocks/slstm/src/cuda_init.py"
xlstm_patch="$PWD/scripts_v8/xlstm-1.0.3-lazy-cuda-init.patch"
xlstm_hash=$(sha256sum "$xlstm_loader")
case "${xlstm_hash%% *}" in
    faa2e8d24d523a12cc4c8b7e6da13f04a66329d94482699f490e509a9270a289)
        patch --forward --force --batch -d "$xlstm_site" -p0 < "$xlstm_patch"
        ;;
    c62be35147ba02a8da1b391e70f215bc4038082b369b31e98171f274225c7555)
        echo 'Local xLSTM compiler-import compatibility patch is already applied.'
        ;;
    *)
        echo 'Refusing to patch an unrecognized xLSTM compiler loader.' >&2
        exit 1
        ;;
esac
xlstm_hash=$(sha256sum "$xlstm_loader")
if [[ "${xlstm_hash%% *}" != c62be35147ba02a8da1b391e70f215bc4038082b369b31e98171f274225c7555 ]]; then
    echo 'Patched xLSTM loader hash verification failed.' >&2
    exit 1
fi
"$xlstm_python" -m pip check
echo 'Installed. To check the local GPU without launching training:'
echo 'OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-xlstm-mixer/bin/python3 scripts_v8/check_phase6_9_xlstm_mixer_environment.py --device cuda --backend vanilla --batch-size 512'
