# xLSTM-Mixer environment compatibility check

The `xlstm-compat` target of `docker/Dockerfile` tests the Phase 6.9 model
before a sandbox or market data bundle exists. It inherits the existing
Python 3.12.3, PyTorch 2.12.1/cu132, and `requirements-container.txt` versions,
then adds `xlstm==1.0.3`, g++, Ninja, and a CUDA development toolkit.
The default final `flowmesh` target keeps the existing runtime dependency set.

The last recorded sandbox used Python 3.12.3, PyTorch `2.12.1+cu132`, and an
RTX PRO 4000 Blackwell (compute capability 12.0). Those records are in the
LWA and TimeDART source manifests. The new sandbox must be checked again.
The owner supplied GPU 0 as `NVIDIA RTX PRO 4000 Blackwell SFF Edition`, UUID
`GPU-eba2dd44-ea94-f656-a4b8-28536cf5dac8`, with 25,655,508,992 bytes of total
memory. Free memory and GPU availability were not supplied. These are expected
hardware values in the xLSTM source manifest; actual availability is measured
when the probe runs.
The Docker image's OS is Debian bookworm; it uses the CUDA 13.2.1 toolkit from
an Ubuntu 22.04 development-image donor. This matches the recorded Python and
PyTorch versions and CUDA major/minor, but does not assert identical OS,
compiler, toolkit patch, host driver, or GPU. The donor tag is listed in
[NVIDIA's supported tags](https://gitlab.com/nvidia/container-images/cuda/-/raw/master/doc/supported-tags.md).

## Build and CPU check

Run from the repository root with a working Docker daemon. On Windows, enable
Docker Desktop's WSL integration for the distro running this repository.

```bash
docker build --platform linux/amd64 \
  --target xlstm-compat --progress plain \
  -f docker/Dockerfile -t fyp-xlstm-compat:phase6-9 .

docker run --rm fyp-xlstm-compat:phase6-9
```

The build installs the exact xLSTM dependency, runs `pip check`, checks the
compiler tools, and exercises the real vanilla backend with forward/backward,
an Adam update, and checkpoint reload. A normal Docker build has no GPU, so
the CUDA extension is compiled at container runtime.

Keep the image local. It inherits the existing bounded FlowMesh reports in
the image; this check does not publish anything to Harbor.

## Vanilla GPU check first, with persistent logs

The host needs a supported NVIDIA driver and Docker GPU access. The image
uses the host GPU and driver through `--gpus all`, as described by
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).

```bash
mkdir -p experiments/phase6_9/xlstm_mixer/feasibility/docker

docker run --rm --gpus all \
  --mount "type=bind,src=$PWD/experiments/phase6_9/xlstm_mixer/feasibility/docker,dst=/results" \
  fyp-xlstm-compat:phase6-9 \
  python scripts_v8/check_phase6_9_xlstm_mixer_environment.py \
    --device cuda --backend vanilla --batch-size 512 \
    --output /results/vanilla-gpu-compatibility.json \
  > experiments/phase6_9/xlstm_mixer/feasibility/docker/vanilla-gpu-smoke.log 2>&1
```

The default backend is now `vanilla`: ordinary PyTorch tensor operations run
on the GPU. This first check does not request custom sLSTM kernel compilation
or enforce nvcc requirements. It exercises batch 512, a one-row final batch,
backward/update, and same-backend checkpoint replay. It records actual
GPU/driver details and compares the observed Python/PyTorch/CUDA and GPU
name/UUID with the expected sandbox record. A hardware mismatch is recorded,
so a successful probe on a local GPU is distinguishable from a sandbox test.

Read the command exit status and JSON `valid` field. Import, forward/backward,
non-finite gradient, and reload failures return a nonzero status. A result
with `cuda_tested: true` and `cuda_extension_tested: false` means vanilla
operations ran on GPU; the custom kernel is still untested.

Local success is evidence about the local GPU. Run the same script in the
new sandbox to check its GPU and driver, including its Blackwell architecture
if that is what gets provisioned:

```bash
.venv/bin/python3 scripts_v8/check_phase6_9_xlstm_mixer_environment.py \
  --device cuda --backend vanilla --batch-size 512 \
  --output experiments/phase6_9/xlstm_mixer/feasibility/sandbox_vanilla_compatibility.json
```

After vanilla succeeds, repeat with `--backend cuda` and a separate result/log
file. That second check requires nvcc to match PyTorch's CUDA major/minor,
compiles the custom kernel, and repeats the batch/backward/reload checks.
The frozen full-training backend remains the pinned custom CUDA backend.

These checks use synthetic tensors and never admit full training. Once both
Stage 1 data bundles exist, the existing
`audit_phase6_9_xlstm_mixer_runtime.py --device cuda --admit-training ...`
must pass with real training batches before `bootstrap... --execute`.

## Rebuild with a different recorded runtime

The toolkit donor, Python base, PyTorch version, and wheel index are build
arguments. Change them together to match an observed sandbox, and retain the
resulting logs and image ID/digest. Do not infer CUDA compatibility from
`nvidia-smi`'s displayed maximum CUDA version alone.

```bash
docker image inspect fyp-xlstm-compat:phase6-9 \
  --format '{{.Id}} {{json .RepoDigests}}'
```

The current tags are versioned references; pin resolved base-image digests
after a successful build for an immutable reconstruction. This target has
not yet been built in the current WSL session because Docker Desktop's WSL
integration is unavailable. The synthetic CPU probe is independently
checkable with the audited dependency checkout.
