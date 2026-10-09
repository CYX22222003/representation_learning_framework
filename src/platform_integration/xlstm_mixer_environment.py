"""Data-free xLSTM-Mixer compatibility probes for Docker and fresh sandboxes."""

from __future__ import annotations

import io
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from baselines.xlstm_mixer import SOURCE_CONTRACT, XLSTMMixer
from training.phase5_encoder import resolve_device, set_seed
from training.phase6_9_xlstm_mixer import (
    PROBE_ATOL,
    PROBE_RTOL,
    runtime_environment_payload,
    smoke_test_xlstm_mixer_training,
    xlstm_dependency_manifest,
)


def _driver_inventory() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def check_cuda_toolchain(environment: dict[str, Any]) -> None:
    """Require nvcc to match the PyTorch CUDA major/minor before compiling."""

    toolchain = environment["compiler_toolchain"]
    for name in ("cxx_path", "cxx_version", "nvcc_path", "nvcc_version"):
        if not toolchain.get(name):
            raise RuntimeError(f"the CUDA extension toolchain is missing {name}")
    if not environment.get("cuda_home"):
        raise RuntimeError("CUDA_HOME must resolve to the installed CUDA toolkit")
    match = re.search(r"release\s+(\d+\.\d+)", toolchain["nvcc_version"])
    if match is None or match.group(1) != environment.get("cuda_version"):
        raise RuntimeError(
            "nvcc and PyTorch must use the same CUDA major/minor: "
            f"nvcc={match.group(1) if match else 'unknown'}, "
            f"torch={environment.get('cuda_version')}"
        )


def synthetic_gpu_smoke(
    device: torch.device, batch_size: int, backend: str = "vanilla"
) -> dict[str, Any]:
    """Update/replay on a GPU using either the vanilla or compiled backend."""

    if device.type != "cuda":
        raise ValueError("synthetic GPU smoke requires a CUDA device")
    if backend not in ("vanilla", "cuda"):
        raise ValueError("backend must be vanilla or cuda")
    if not 1 <= batch_size <= 512:
        raise ValueError("batch size must be in [1,512]")
    capability = torch.cuda.get_device_capability(device)
    if backend == "cuda" and capability < (8, 0):
        raise RuntimeError("xlstm==1.0.3 CUDA sLSTM requires compute capability >= 8.0")
    torch.cuda.set_device(
        torch.cuda.current_device() if device.index is None else device.index
    )
    set_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = XLSTMMixer(backend=backend).to(device)
    torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - started
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    batch_results = []
    # drop_last=False requires the admitted physical batch and a smaller last
    # batch to work with the same constructed backend on the GPU.
    sizes = [batch_size] if batch_size == 1 else [batch_size, 1]
    for size in sizes:
        contexts = torch.rand(size, 64, 5, device=device, dtype=torch.float32)
        targets = torch.rand(size, 8, 5, device=device, dtype=torch.float32)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = model(contexts)
        loss = model.full_path_l1_loss(prediction, targets)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"missing/non-finite CUDA gradient: {name}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite CUDA gradient norm")
        optimizer.step()
        torch.cuda.synchronize(device)
        batch_results.append(
            {
                "batch_size": size,
                "prediction_shape": list(prediction.shape),
                "loss": float(loss.detach()),
                "gradients_finite": True,
                "gradient_norm_before_clip": float(norm),
                "step_seconds": time.perf_counter() - started,
            }
        )
    peak_allocation = torch.cuda.max_memory_allocated(device)
    probe = torch.rand(2, 64, 5, device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        expected = model(probe).cpu()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    reloaded = XLSTMMixer(backend=backend).to(device).eval()
    reloaded.load_state_dict(torch.load(stream, map_location=device, weights_only=True))
    with torch.no_grad():
        actual = reloaded(probe).cpu()
    torch.testing.assert_close(actual, expected, rtol=PROBE_RTOL, atol=PROBE_ATOL)
    return {
        "valid": True,
        "backend": backend,
        "model_construction_seconds": build_seconds,
        "cuda_extension_build_seconds": build_seconds if backend == "cuda" else None,
        "cuda_extension_tested": backend == "cuda",
        "batches": batch_results,
        "peak_cuda_memory_bytes": int(peak_allocation),
        "same_backend_checkpoint_replay": True,
        "replay_tolerances": {"rtol": PROBE_RTOL, "atol": PROBE_ATOL},
    }


def check_xlstm_mixer_environment(
    device: str = "cpu", batch_size: int = 512, backend: str = "vanilla"
) -> dict[str, Any]:
    resolved = resolve_device(device)
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError("compatibility checks support only CPU and CUDA")
    if not 1 <= batch_size <= 512:
        raise ValueError("batch size must be in [1,512]")
    if backend not in ("vanilla", "cuda"):
        raise ValueError("backend must be vanilla or cuda")
    if resolved.type == "cpu" and backend != "vanilla":
        raise ValueError("CPU compatibility checks require the vanilla backend")
    environment = runtime_environment_payload(resolved)
    source_path = (
        Path(__file__).resolve().parents[2]
        / "docs/baselines/xLSTM-Mixer/source_manifest.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = source["compatibility_test_expectations"]
    observed_gpu = None
    if resolved.type == "cuda":
        properties = torch.cuda.get_device_properties(resolved)
        uuid = getattr(properties, "uuid", None)
        observed_gpu = {
            "index": torch.cuda.current_device() if resolved.index is None else resolved.index,
            "name": properties.name,
            "uuid": str(uuid) if uuid is not None else None,
            "memory_total_bytes": int(properties.total_memory),
            "memory_free_bytes": int(torch.cuda.mem_get_info(resolved)[0]),
            "gpu_available": True,
        }
    result = {
        "schema_version": "phase6-9-xlstm-mixer-environment-check-v2",
        "device": str(resolved),
        "backend": backend,
        "environment": environment,
        "expected_sandbox_environment": expected,
        "observed_gpu": observed_gpu,
        "matches_expected": {
            "python": environment["python"].split()[0] == expected["python"],
            "pytorch": environment["torch"] == expected["pytorch"],
            "cuda": environment["cuda_version"] == expected["cuda"],
            "gpu_name": observed_gpu["name"] == expected["gpu"]["name"] if observed_gpu else None,
            "gpu_uuid": observed_gpu["uuid"] == expected["gpu"]["uuid"] if observed_gpu else None,
        },
        "driver_inventory": _driver_inventory(),
        "dependency": xlstm_dependency_manifest(),
        "source_contract_sha256": SOURCE_CONTRACT.sha256,
        "cpu_vanilla_smoke": smoke_test_xlstm_mixer_training(),
        "gpu_smoke": None,
        "cuda_tested": False,
        "cuda_extension_tested": False,
        "data_required": False,
        "training_admitted": False,
        "training_launched": False,
    }
    if resolved.type == "cuda":
        if backend == "cuda":
            check_cuda_toolchain(environment)
        result["gpu_smoke"] = synthetic_gpu_smoke(resolved, batch_size, backend)
        result["cuda_tested"] = True
        result["cuda_extension_tested"] = backend == "cuda"
    result["valid"] = True
    return result
