"""Shared, non-entry-point utilities for Phase 3 encoder experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PHASE3_DATA_ROOT = ROOT / "data" / "phase3"
PHASE3_EXPERIMENT_ROOT = ROOT / "experiments" / "phase3"
DEFAULT_PROCESSED_NPZ = PHASE3_DATA_ROOT / "processed" / "market_4h_seq64_top50.npz"
DEFAULT_ENCODER_ROOT = PHASE3_EXPERIMENT_ROOT / "encoder_pretraining"
EPOCH_BUDGETS = (5, 15, 50)
SEED = 0
ENCODER_VARIANTS = (
    "vae_mlp",
    "contrastive_cnn",
    "contrastive_lstm",
    "contrastive_transformer",
    "byol_cnn",
    "byol_lstm",
    "byol_transformer",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_phase3_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    required = root.resolve()
    if resolved != required and required not in resolved.parents:
        raise ValueError(f"path must be under {required}: {resolved}")
    if any(part.endswith("_old") for part in resolved.parts):
        raise ValueError(f"legacy _old path is prohibited: {resolved}")
    return resolved


def refuse_occupied(paths: Iterable[Path]) -> None:
    occupied = []
    for path in paths:
        if path.is_dir() and any(path.iterdir()):
            occupied.append(str(path))
        elif path.is_file():
            occupied.append(str(path))
    if occupied:
        raise FileExistsError(f"refusing to overwrite existing Phase 3 artifacts: {occupied}")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_manifest(device: torch.device) -> dict[str, Any]:
    return {
        "python": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "git_commit": git_commit(),
    }
