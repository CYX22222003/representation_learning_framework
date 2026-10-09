"""Gated, resumable Phase 6.9 xLSTM-Mixer training and replay.

The module deliberately does not construct the future-path dataset. It accepts
only the separately validated Stage-1 artifact and refuses training until a
matching Stage-3 CUDA admission manifest exists.
"""

from __future__ import annotations

import io
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from baselines.xlstm_mixer import (
    METHOD_ID,
    SOURCE_CONTRACT,
    XLSTMMixer,
    XLSTMMixerConfig,
)
from data_processing.phase5_walks import sha256_arrays, sha256_file
from training.phase5_absolute_price import relative_skill
from training.phase5_downstream import regression_metrics
from training.phase5_encoder import environment_manifest, resolve_device, set_seed, write_json


METHOD = "xm_mv8"
REPORTING_LABEL = "xLSTM-Mixer (XM-MV8)"
SNAPSHOT_EPOCHS = (5, 15, 50)
DATA_SCHEMA_VERSION = "phase6-9-xlstm-mixer-data-v1"
TRAINING_SCHEMA_VERSION = "phase6-9-xlstm-mixer-training-v1"
ADMISSION_SCHEMA_VERSION = "phase6-9-xlstm-mixer-cuda-admission-v1"
MATRIX_SCHEMA_VERSION = "phase6-9-xlstm-mixer-training-matrix-v1"
REPLAY_SCHEMA_VERSION = "phase6-9-xlstm-mixer-replay-v1"
PROBE_RTOL = 1e-6
PROBE_ATOL = 1e-7
NATIVE_HOUR_NS = 3_600_000_000_000
EXPECTED_ROWS = {
    1: {"train": 32_470, "evaluation": 27_786},
    2: {"train": 53_112, "evaluation": 12_115},
}
IDENTITY_FIELDS = (
    "row_ids",
    "condition_ids",
    "segment_ids",
    "decision_time_ns",
    "decision_availability_ns",
    "target_time_ns",
    "target_availability_ns",
)


@dataclass(frozen=True)
class Phase69XLSTMMixerConfig:
    walk: int
    epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    seed: int = 0
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    device: str = "cuda"
    backend: str = "vanilla"

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_epochs", tuple(self.snapshot_epochs))
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.epochs != 50 or self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("XM-MV8 requires 50 epochs and 5/15/50 snapshots")
        if self.seed != 0:
            raise ValueError("XM-MV8 is frozen to seed 0")
        if self.batch_size <= 0 or self.batch_size > 512:
            raise ValueError("XM-MV8 physical batch size must be in [1,512]")
        if self.learning_rate != 1e-4 or self.weight_decay != 0.0:
            raise ValueError("XM-MV8 optimizer settings differ from the freeze")
        if self.gradient_clip_norm != 1.0:
            raise ValueError("XM-MV8 gradient clipping must use norm 1.0")
        if self.backend not in ("cuda", "vanilla"):
            raise ValueError("XM-MV8 backend must be cuda or vanilla")
        if not self.device.startswith("cuda"):
            raise ValueError("full XM-MV8 training requires a CUDA device")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


@dataclass(frozen=True)
class XLSTMMixerDataset:
    train_contexts: np.ndarray
    train_targets: np.ndarray
    evaluation_contexts: np.ndarray
    evaluation_targets: np.ndarray
    train_metadata: dict[str, np.ndarray]
    evaluation_metadata: dict[str, np.ndarray]
    manifest: dict[str, Any]
    dataset_sha256: str
    manifest_sha256: str
    train_identity_sha256: str
    evaluation_identity_sha256: str


def phase_data_path(root: Path, walk: int) -> Path:
    return (
        Path(root)
        / "experiments"
        / "phase6_9"
        / "xlstm_mixer"
        / "data"
        / "absolute_price_h8_multivariate"
        / f"walk{walk}.npz"
    )


def phase_run_root(root: Path, walk: int) -> Path:
    return (
        Path(root)
        / "experiments"
        / "phase6_9"
        / "xlstm_mixer"
        / "downstream"
        / "absolute_price_h8"
        / f"walk{walk}"
        / METHOD
        / "seed0"
    )


def phase_admission_path(root: Path) -> Path:
    return (
        Path(root)
        / "experiments"
        / "phase6_9"
        / "xlstm_mixer"
        / "feasibility"
        / "cuda_admission.json"
    )


def _data_manifest_path(dataset_path: Path) -> Path:
    return Path(f"{dataset_path}.manifest.json")


def _identity_hash(metadata: Mapping[str, np.ndarray]) -> str:
    return sha256_arrays(*(np.asarray(metadata[field]) for field in IDENTITY_FIELDS))


def _validate_split(
    split: str,
    contexts: np.ndarray,
    targets: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    *,
    expected_rows: int | None,
) -> str:
    if contexts.dtype != np.float32 or contexts.ndim != 3 or contexts.shape[1:] != (64, 5):
        raise ValueError(f"{split}_contexts must be float32 [N,64,5]")
    if targets.dtype != np.float32 or targets.shape != (len(contexts), 8, 5):
        raise ValueError(f"{split}_targets must be float32 [N,8,5]")
    if expected_rows is not None and len(contexts) != expected_rows:
        raise ValueError(f"{split} row count differs from the frozen intersection")
    if not np.isfinite(contexts).all() or not np.isfinite(targets).all():
        raise ValueError(f"{split} contexts/targets must be finite")
    if np.any((contexts[:, :, :4] < 0.0) | (contexts[:, :, :4] > 1.0)):
        raise ValueError(f"{split} context OHLC must remain in [0,1]")
    if np.any((targets[:, :, :4] < 0.0) | (targets[:, :, :4] > 1.0)):
        raise ValueError(f"{split} target OHLC must remain in [0,1]")
    for name, values in (("context", contexts), ("target", targets)):
        open_ = values[:, :, 0]
        high = values[:, :, 1]
        low = values[:, :, 2]
        close = values[:, :, 3]
        if np.any(
            (high < np.maximum.reduce((open_, low, close)))
            | (low > np.minimum.reduce((open_, high, close)))
        ):
            raise ValueError(f"{split} {name} OHLC ordering is invalid")

    count = len(contexts)
    for field in IDENTITY_FIELDS:
        if field not in metadata:
            raise ValueError(f"{split} metadata is missing {field}")
    one_dimensional = (
        "row_ids",
        "condition_ids",
        "segment_ids",
        "decision_time_ns",
        "decision_availability_ns",
    )
    for field in one_dimensional:
        if np.asarray(metadata[field]).shape != (count,):
            raise ValueError(f"{split}_{field} must have shape [N]")
    for field in ("target_time_ns", "target_availability_ns"):
        if np.asarray(metadata[field]).shape != (count, 8):
            raise ValueError(f"{split}_{field} must have shape [N,8]")
    for field in ("target_observed", "target_imputed"):
        if field not in metadata or np.asarray(metadata[field]).shape != (count, 8):
            raise ValueError(f"{split}_{field} must have shape [N,8]")

    row_ids = np.asarray(metadata["row_ids"])
    if len(np.unique(row_ids)) != count:
        raise ValueError(f"{split} row identities must be unique")
    decision = np.asarray(metadata["decision_time_ns"], dtype=np.int64)
    decision_availability = np.asarray(
        metadata["decision_availability_ns"], dtype=np.int64
    )
    target_time = np.asarray(metadata["target_time_ns"], dtype=np.int64)
    target_availability = np.asarray(
        metadata["target_availability_ns"], dtype=np.int64
    )
    expected_time = decision[:, None] + NATIVE_HOUR_NS * np.arange(1, 9)[None, :]
    if not np.array_equal(target_time, expected_time):
        raise ValueError(f"{split} targets are not the exact next eight hourly bars")
    if not np.array_equal(decision_availability, decision + NATIVE_HOUR_NS):
        raise ValueError(f"{split} decision availability is inconsistent")
    if not np.array_equal(target_availability, target_time + NATIVE_HOUR_NS):
        raise ValueError(f"{split} target availability is inconsistent")
    if not np.asarray(metadata["target_observed"], dtype=bool).all():
        raise ValueError(f"{split} contains a non-observed target bar")
    if np.asarray(metadata["target_imputed"], dtype=bool).any():
        raise ValueError(f"{split} contains an imputed target bar")
    if not np.array_equal(contexts[:, -1, 3], np.asarray(metadata["current_close"], dtype=np.float32)):
        raise ValueError(f"{split} current close does not replay from the context")
    if not np.array_equal(targets[:, -1, 3], np.asarray(metadata["target_close"], dtype=np.float32)):
        raise ValueError(f"{split} target close does not replay from the path")
    return _identity_hash(metadata)


def validate_xlstm_mixer_arrays(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    *,
    enforce_frozen_counts: bool = True,
) -> dict[str, Any]:
    """Validate the training-facing Stage-1 bundle without fitting anything."""

    if manifest.get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("unsupported XM-MV8 data schema")
    if manifest.get("method_id") != METHOD_ID or manifest.get("method") != METHOD:
        raise ValueError("data manifest has the wrong method identity")
    walk = int(manifest.get("walk", -1))
    if walk not in (1, 2):
        raise ValueError("data manifest walk must be 1 or 2")
    if manifest.get("validation_passed") is not True:
        raise ValueError("Stage-1 standalone data validation has not passed")
    policy = manifest.get("full_path_policy", {})
    expected_policy = {
        "horizon_hours": list(range(1, 9)),
        "same_condition": True,
        "same_segment": True,
        "all_target_bars_observed": True,
        "target_imputation_allowed": False,
        "availability_only_intersection": True,
    }
    if policy != expected_policy:
        raise ValueError("full-path availability policy differs from the freeze")
    if manifest.get("channel_order") != ["open", "high", "low", "close", "volume"]:
        raise ValueError("XM-MV8 channel order differs from the freeze")
    if manifest.get("additional_xlstm_channel_scaler") is not False:
        raise ValueError("an xLSTM-specific channel scaler is prohibited")
    volume = manifest.get("volume_transform", {})
    if not volume.get("sha256") or not all(name in volume for name in ("mean", "denominator")):
        raise ValueError("upstream walk-training volume transform provenance is incomplete")

    required_arrays = {
        f"{split}_{field}"
        for split in ("train", "evaluation")
        for field in (
            "contexts",
            "targets",
            *IDENTITY_FIELDS,
            "target_observed",
            "target_imputed",
            "current_close",
            "target_close",
        )
    }
    missing = sorted(required_arrays - set(arrays))
    if missing:
        raise ValueError(f"XM-MV8 data bundle is missing arrays: {missing}")

    records: dict[str, Any] = {}
    row_counts = manifest.get("row_counts", {})
    identity_hashes = manifest.get("identity_hashes", {})
    for split in ("train", "evaluation"):
        metadata = {
            field: np.asarray(arrays[f"{split}_{field}"])
            for field in (
                *IDENTITY_FIELDS,
                "target_observed",
                "target_imputed",
                "current_close",
                "target_close",
            )
        }
        contexts = np.asarray(arrays[f"{split}_contexts"])
        targets = np.asarray(arrays[f"{split}_targets"])
        expected_rows = EXPECTED_ROWS[walk][split] if enforce_frozen_counts else None
        identity = _validate_split(
            split,
            contexts,
            targets,
            metadata,
            expected_rows=expected_rows,
        )
        if int(row_counts.get(split, -1)) != len(contexts):
            raise ValueError(f"{split} row count disagrees with the manifest")
        if identity_hashes.get(split) != identity:
            raise ValueError(f"{split} identity hash disagrees with the manifest")
        records[split] = {"rows": len(contexts), "identity_sha256": identity}

    train_rows = np.asarray(arrays["train_row_ids"])
    evaluation_rows = np.asarray(arrays["evaluation_row_ids"])
    if np.intersect1d(train_rows, evaluation_rows).size:
        raise ValueError("training and evaluation row identities overlap")
    cutoff = int(manifest["training_cutoff_ns"])
    evaluation_start = int(manifest["evaluation_start_ns"])
    evaluation_end = int(manifest["evaluation_end_ns"])
    if cutoff != evaluation_start:
        raise ValueError("evaluation must begin at the walk training cutoff")
    if np.any(np.asarray(arrays["train_target_availability_ns"]) >= cutoff):
        raise ValueError("a training target matures at or after the cutoff")
    evaluation_decisions = np.asarray(arrays["evaluation_decision_availability_ns"])
    evaluation_targets = np.asarray(arrays["evaluation_target_availability_ns"])
    if np.any(evaluation_decisions < evaluation_start):
        raise ValueError("an evaluation decision precedes the evaluation interval")
    if np.any(evaluation_targets >= evaluation_end):
        raise ValueError("an evaluation target reaches beyond the evaluation interval")
    return {"valid": True, "walk": walk, **records}


def load_xlstm_mixer_data(
    dataset_path: Path,
    walk: int,
    *,
    enforce_frozen_counts: bool = True,
) -> XLSTMMixerDataset:
    dataset_path = Path(dataset_path)
    manifest_path = _data_manifest_path(dataset_path)
    if not dataset_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("XM-MV8 data bundle or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("walk", -1)) != walk:
        raise ValueError("XM-MV8 dataset walk mismatch")
    dataset_hash = sha256_file(dataset_path)
    manifest_hash = sha256_file(manifest_path)
    if manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("XM-MV8 dataset hash disagrees with the manifest")
    with np.load(dataset_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    validation = validate_xlstm_mixer_arrays(
        arrays, manifest, enforce_frozen_counts=enforce_frozen_counts
    )
    train_metadata = {
        name.removeprefix("train_"): values
        for name, values in arrays.items()
        if name.startswith("train_") and name not in {"train_contexts", "train_targets"}
    }
    evaluation_metadata = {
        name.removeprefix("evaluation_"): values
        for name, values in arrays.items()
        if name.startswith("evaluation_")
        and name not in {"evaluation_contexts", "evaluation_targets"}
    }
    return XLSTMMixerDataset(
        train_contexts=np.asarray(arrays["train_contexts"], dtype=np.float32),
        train_targets=np.asarray(arrays["train_targets"], dtype=np.float32),
        evaluation_contexts=np.asarray(arrays["evaluation_contexts"], dtype=np.float32),
        evaluation_targets=np.asarray(arrays["evaluation_targets"], dtype=np.float32),
        train_metadata=train_metadata,
        evaluation_metadata=evaluation_metadata,
        manifest=manifest,
        dataset_sha256=dataset_hash,
        manifest_sha256=manifest_hash,
        train_identity_sha256=validation["train"]["identity_sha256"],
        evaluation_identity_sha256=validation["evaluation"]["identity_sha256"],
    )


def implementation_fingerprint(root: Path | None = None) -> str:
    repository = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    relative_paths = (
        "src/baselines/xlstm_mixer/config.py",
        "src/baselines/xlstm_mixer/normalization.py",
        "src/baselines/xlstm_mixer/backend.py",
        "src/baselines/xlstm_mixer/model.py",
        "src/training/phase6_9_xlstm_mixer.py",
        "src/data_processing/phase6_9_xlstm_mixer.py",
        "scripts_v8/prepare_phase6_9_xlstm_mixer_data.py",
        "scripts_v8/validate_phase6_9_xlstm_mixer_data.py",
        "scripts_v8/audit_phase6_9_xlstm_mixer_runtime.py",
        "scripts_v8/bootstrap_phase6_9_xlstm_mixer.py",
        "scripts_v8/validate_phase6_9_xlstm_mixer.py",
    )
    digest = sha256()
    for relative in relative_paths:
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"XM-MV8 implementation file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def xlstm_dependency_manifest() -> dict[str, Any]:
    import xlstm

    if getattr(xlstm, "__version__", None) != SOURCE_CONTRACT.xlstm_version:
        raise RuntimeError("runtime xlstm version differs from the approved pin")
    package_root = Path(xlstm.__file__).resolve().parent
    digest = sha256()
    source_files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".cu", ".cc", ".h", ".cuh"}
    )
    for path in source_files:
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "version": str(xlstm.__version__),
        "module_path": str(Path(xlstm.__file__).resolve()),
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(source_files),
        "license": SOURCE_CONTRACT.xlstm_license,
        "pinned_commit": SOURCE_CONTRACT.xlstm_commit,
    }


def _dependency_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    if "module_path" in values:
        values["module_path"] = str(values["module_path"]).casefold()
    return values


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy[0],
            "keys": torch.from_numpy(numpy[1].copy()),
            "position": int(numpy[2]),
            "has_gauss": int(numpy[3]),
            "cached_gaussian": float(numpy[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "sampler_generator": generator.get_state(),
    }


def _restore_rng_state(state: Mapping[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state(
        (
            str(numpy["name"]),
            np.asarray(numpy["keys"].cpu(), dtype=np.uint32),
            int(numpy["position"]),
            int(numpy["has_gauss"]),
            float(numpy["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = list(state.get("torch_cuda", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("resume checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    generator.set_state(state["sampler_generator"].cpu())


def shuffled_batches(
    row_count: int, batch_size: int, generator: torch.Generator
) -> tuple[list[torch.Tensor], int]:
    if row_count <= 0 or batch_size <= 0:
        raise ValueError("row count and batch size must be positive")
    batches = list(torch.randperm(row_count, generator=generator).split(batch_size))
    return batches, int(len(batches[-1]))


def _assert_finite_gradients(model: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("XM-MV8 update produced no gradients")
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("XM-MV8 update produced non-finite gradients")


@torch.no_grad()
def _probe_output(model: XLSTMMixer, raw_probe: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    output = model(raw_probe.to(next(model.parameters()).device)).cpu()
    model.train(was_training)
    return output


def smoke_test_xlstm_mixer_training() -> dict[str, Any]:
    """Bounded vanilla-backend shape/backward/checkpoint smoke."""

    set_seed(0)
    model = XLSTMMixer(backend="vanilla")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    contexts = torch.randn(2, 64, 5)
    targets = torch.randn(2, 8, 5)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(contexts)
    loss = model.full_path_l1_loss(prediction, targets)
    loss.backward()
    _assert_finite_gradients(model)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    expected = _probe_output(model, contexts)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    replay = XLSTMMixer(backend="vanilla")
    replay.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
    actual = _probe_output(replay, contexts)
    if not torch.allclose(actual, expected, rtol=PROBE_RTOL, atol=PROBE_ATOL):
        raise RuntimeError("vanilla checkpoint replay failed")
    return {
        "valid": True,
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach()),
        "gradient_norm_before_clip": float(norm),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_replay": True,
    }


def resource_smoke_xlstm_mixer(
    dataset_path: Path,
    *,
    walk: int,
    batch_size: int,
    device: str = "cuda",
    backend: str = "vanilla",
) -> dict[str, Any]:
    data = load_xlstm_mixer_data(dataset_path, walk)
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("XM-MV8 resource admission requires CUDA")
    set_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(resolved)
    build_started = time.perf_counter()
    model = XLSTMMixer(backend=backend).to(resolved)
    model_build_seconds = time.perf_counter() - build_started
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    contexts = torch.from_numpy(data.train_contexts[:batch_size]).to(resolved)
    targets = torch.from_numpy(data.train_targets[:batch_size]).to(resolved)
    if len(contexts) != batch_size:
        raise ValueError("resource smoke could not form the admitted physical batch")
    torch.cuda.synchronize(resolved)
    step_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(contexts)
    loss = model.full_path_l1_loss(prediction, targets)
    loss.backward()
    _assert_finite_gradients(model)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # drop_last=False: a one-row final batch must use the same backend safely.
    optimizer.zero_grad(set_to_none=True)
    remainder_loss = model.full_path_l1_loss(model(contexts[:1]), targets[:1])
    remainder_loss.backward()
    _assert_finite_gradients(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(resolved)
    step_seconds = time.perf_counter() - step_started
    training_peak = int(torch.cuda.max_memory_allocated(resolved))

    raw_probe = contexts[: min(4, len(contexts))].detach().cpu()
    expected = _probe_output(model, raw_probe)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    replay = XLSTMMixer(backend=backend).to(resolved)
    replay.load_state_dict(torch.load(stream, map_location=resolved, weights_only=True))
    actual = _probe_output(replay, raw_probe)
    replay_ok = bool(torch.allclose(actual, expected, rtol=PROBE_RTOL, atol=PROBE_ATOL))
    if not replay_ok:
        raise RuntimeError("same-CUDA-backend checkpoint replay failed")
    return {
        "walk": walk,
        "dataset_sha256": data.dataset_sha256,
        "manifest_sha256": data.manifest_sha256,
        "train_identity_sha256": data.train_identity_sha256,
        "evaluation_identity_sha256": data.evaluation_identity_sha256,
        "physical_batch_size": batch_size,
        "backend": backend,
        "one_row_remainder_passed": bool(torch.isfinite(remainder_loss)),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": True,
        "gradient_norm_before_clip": float(gradient_norm),
        "prediction_shape": list(prediction.shape),
        "model_build_seconds": model_build_seconds,
        "step_seconds": step_seconds,
        "peak_cuda_memory_bytes": training_peak,
        "same_backend_checkpoint_replay": replay_ok,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "environment": json.loads(json.dumps(environment_manifest(resolved))),
    }


def validate_runtime_admission(
    admission_path: Path,
    dataset_path: Path,
    config: Phase69XLSTMMixerConfig,
) -> dict[str, Any]:
    admission_path = Path(admission_path)
    if not admission_path.is_file():
        raise FileNotFoundError("XM-MV8 CUDA admission manifest is missing")
    payload = json.loads(admission_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ADMISSION_SCHEMA_VERSION:
        raise ValueError("XM-MV8 CUDA admission schema mismatch")
    if payload.get("method") != METHOD or payload.get("admitted") is not True:
        raise ValueError("XM-MV8 CUDA training was not admitted")
    if payload.get("backend") != config.backend:
        raise ValueError("XM-MV8 admitted backend mismatch")
    if int(payload.get("physical_batch_size", -1)) != config.batch_size:
        raise ValueError("XM-MV8 admitted physical batch size mismatch")
    if payload.get("source_contract_sha256") != SOURCE_CONTRACT.sha256:
        raise ValueError("XM-MV8 admitted source contract drift")
    if payload.get("implementation_sha256") != implementation_fingerprint():
        raise ValueError("XM-MV8 implementation changed after CUDA admission")
    if _dependency_signature(payload.get("dependency", {})) != _dependency_signature(xlstm_dependency_manifest()):
        raise ValueError("XM-MV8 dependency changed after admission")
    current = runtime_environment_payload(resolve_device(config.device))
    stable_fields = ("python", "torch", "numpy", "cuda_available", "cuda_version",
                     "device", "device_name", "cuda_compute_capability",
                     "torch_build_configuration", "runtime_package_versions")
    if any(payload.get("environment", {}).get(key) != current.get(key) for key in stable_fields):
        raise ValueError("XM-MV8 runtime changed after admission")
    entry = next(
        (row for row in payload.get("walks", []) if int(row.get("walk", -1)) == config.walk),
        None,
    )
    if entry is None:
        raise ValueError("XM-MV8 admission has no matching walk")
    data = load_xlstm_mixer_data(dataset_path, config.walk)
    expected = {
        "dataset_sha256": data.dataset_sha256,
        "manifest_sha256": data.manifest_sha256,
        "train_identity_sha256": data.train_identity_sha256,
        "evaluation_identity_sha256": data.evaluation_identity_sha256,
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        raise ValueError("XM-MV8 admitted data identity mismatch")
    if entry.get("same_backend_checkpoint_replay") is not True:
        raise ValueError("XM-MV8 admitted replay gate is missing")
    limits = payload.get("resource_limits", {})
    memory = float(limits.get("max_peak_memory_gib", 0)) * 1024**3
    seconds = float(limits.get("max_smoke_seconds", 0))
    if not (math.isfinite(memory) and memory > 0 and math.isfinite(seconds) and seconds > 0):
        raise ValueError("XM-MV8 admitted resource limits are invalid")
    if (entry.get("loss_finite") is not True or entry.get("gradients_finite") is not True
        or entry.get("one_row_remainder_passed") is not True
        or entry.get("backend") != config.backend
        or not math.isfinite(float(entry.get("loss", float("nan"))))
        or not 0 <= float(entry.get("peak_cuda_memory_bytes", float("inf"))) <= memory
        or not 0 <= float(entry.get("model_build_seconds", float("inf")))
                    + float(entry.get("step_seconds", float("inf"))) <= seconds):
        raise ValueError("XM-MV8 admitted resource probe is invalid")
    return payload


def _config_from_payload(payload: Mapping[str, Any], device: str) -> Phase69XLSTMMixerConfig:
    values = dict(payload)
    values["snapshot_epochs"] = tuple(values["snapshot_epochs"])
    values["device"] = device
    return Phase69XLSTMMixerConfig(**values)


def _history_arrays(history: list[Mapping[str, float]]) -> dict[str, np.ndarray]:
    fields = sorted({field for row in history for field in row})
    return {
        "epochs": np.arange(1, len(history) + 1, dtype=np.int32),
        **{
            field: np.asarray([row[field] for row in history], dtype=np.float64)
            for field in fields
        },
    }


def _checkpoint_payload(
    *,
    model: XLSTMMixer,
    optimizer: torch.optim.Optimizer,
    config: Phase69XLSTMMixerConfig,
    data: XLSTMMixerDataset,
    admission_path: Path,
    completed_epoch: int,
    history: list[Mapping[str, float]],
    generator: torch.Generator,
    raw_probe: torch.Tensor,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "phase": "6.9",
        "method": METHOD,
        "method_id": METHOD_ID,
        "walk": config.walk,
        "completed_epoch": completed_epoch,
        "config": config.to_dict(),
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset_sha256": data.dataset_sha256,
        "data_manifest_sha256": data.manifest_sha256,
        "train_identity_sha256": data.train_identity_sha256,
        "evaluation_identity_sha256": data.evaluation_identity_sha256,
        "runtime_admission_sha256": sha256_file(admission_path),
        "implementation_sha256": implementation_fingerprint(),
        "source_contract_sha256": SOURCE_CONTRACT.sha256,
        "history": [dict(row) for row in history],
        "rng_states": _capture_rng_state(generator),
        "sampler_state": {"next_epoch": completed_epoch + 1, "batch_position": 0},
        "probe": {
            "raw_input": raw_probe.cpu(),
            "forecast": _probe_output(model, raw_probe),
        },
        "code_revision": environment.get("git_commit"),
        "device_metadata": dict(environment),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(next(model.parameters()).device))
            if next(model.parameters()).device.type == "cuda"
            else 0
        ),
    }


@torch.no_grad()
def _predict_full_path(
    model: XLSTMMixer,
    contexts: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    values = torch.from_numpy(np.asarray(contexts, dtype=np.float32))
    outputs = [model(batch.to(device)).cpu().numpy() for batch in values.split(batch_size)]
    result = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if result.shape != (len(contexts), 8, 5) or not np.isfinite(result).all():
        raise FloatingPointError("XM-MV8 produced invalid full-path predictions")
    return result


def _full_path_diagnostics(
    prediction: np.ndarray,
    targets: np.ndarray,
    volume_transform: Mapping[str, Any],
) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    channel_names = ("open", "high", "low", "close", "volume")
    per_channel = {
        name: regression_metrics(pred[:, :, index], target[:, :, index])
        for index, name in enumerate(channel_names)
    }
    probability_invalid = np.any((pred[:, :, :4] < 0.0) | (pred[:, :, :4] > 1.0), axis=(1, 2))
    high = pred[:, :, 1]
    low = pred[:, :, 2]
    open_ = pred[:, :, 0]
    close = pred[:, :, 3]
    ohlc_invalid = np.any(
        (high < np.maximum.reduce((open_, low, close)))
        | (low > np.minimum.reduce((open_, high, close))),
        axis=1,
    )
    raw_volume = (
        pred[:, :, 4] * float(volume_transform["denominator"])
        + float(volume_transform["mean"])
    )
    volume_invalid = np.any(raw_volume < 0.0, axis=1)
    return {
        "accepted_unit_full_path_mae": float(np.mean(np.abs(pred - target))),
        "accepted_unit_full_path_mse": float(np.mean(np.square(pred - target))),
        "per_channel": per_channel,
        "invalid_probability_path_fraction": float(np.mean(probability_invalid)),
        "invalid_ohlc_path_fraction": float(np.mean(ohlc_invalid)),
        "negative_raw_volume_path_fraction": float(np.mean(volume_invalid)),
    }


def evaluate_xlstm_predictions(
    prediction: np.ndarray,
    data: XLSTMMixerDataset,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float]]:
    predicted_price = np.asarray(prediction[:, -1, 3], dtype=np.float64)
    target_price = np.asarray(data.evaluation_metadata["target_close"], dtype=np.float64)
    current = np.asarray(data.evaluation_metadata["current_close"], dtype=np.float64)
    price = regression_metrics(predicted_price, target_price)
    persistence = regression_metrics(current, target_price)
    predicted_delta = predicted_price - current
    target_delta = target_price - current
    movement = regression_metrics(predicted_delta, target_delta)
    last_hour_reversal = -(
        np.asarray(data.evaluation_contexts[:, -1, 3], dtype=np.float64)
        - np.asarray(data.evaluation_contexts[:, -2, 3], dtype=np.float64)
    )
    reversal = regression_metrics(last_hour_reversal, target_delta)
    condition_ids = np.asarray(data.evaluation_metadata["condition_ids"])
    per_contract = {
        str(condition): {
            "price": regression_metrics(
                predicted_price[condition_ids == condition],
                target_price[condition_ids == condition],
            ),
            "implied_movement": regression_metrics(
                predicted_delta[condition_ids == condition],
                target_delta[condition_ids == condition],
            ),
        }
        for condition in np.unique(condition_ids)
    }
    macro_metrics = {}
    for metric in ("mae", "rmse", "mse", "pearson", "spearman"):
        values = [
            float(record["price"][metric])
            for record in per_contract.values()
            if math.isfinite(float(record["price"][metric]))
        ]
        macro_metrics[metric] = float(np.mean(values)) if values else float("nan")
    decision_times = np.asarray(data.evaluation_metadata["decision_time_ns"])
    timestamp_rank_ic = []
    for timestamp in np.unique(decision_times):
        selected = decision_times == timestamp
        if np.count_nonzero(selected) < 2:
            continue
        value = float(
            regression_metrics(predicted_delta[selected], target_delta[selected])["spearman"]
        )
        if math.isfinite(value):
            timestamp_rank_ic.append(value)
    diagnostics = _full_path_diagnostics(
        prediction,
        data.evaluation_targets,
        data.manifest["volume_transform"],
    )
    payload = {
        "price": price,
        "persistence_reference": persistence,
        "persistence_relative_skill": relative_skill(price, persistence),
        "implied_movement": movement,
        "last_hour_reversal": reversal,
        "per_contract": per_contract,
        "contract_macro_price": macro_metrics,
        "timestamp_cross_sectional_rank_ic": {
            "timestamp_count": len(timestamp_rank_ic),
            "mean": (
                float(np.mean(timestamp_rank_ic)) if timestamp_rank_ic else float("nan")
            ),
            "median": (
                float(np.median(timestamp_rank_ic)) if timestamp_rank_ic else float("nan")
            ),
        },
        "full_path_diagnostics": diagnostics,
        "extra_supervision_disclosure": (
            "trained on all five channels at horizons t+1..t+8; headline is close[t+8]"
        ),
    }
    arrays = {
        "prediction_full_path": np.asarray(prediction, dtype=np.float32),
        "target_full_path": np.asarray(data.evaluation_targets, dtype=np.float32),
        "prediction_future_price": predicted_price,
        "target_future_price": target_price,
        "current_close": current,
        "prediction_delta_h8": predicted_delta,
        "target_delta_h8": target_delta,
        "last_hour_reversal": last_hour_reversal,
        **{
            name: np.asarray(values)
            for name, values in data.evaluation_metadata.items()
        },
    }
    summary = {
        "mae": float(price["mae"]),
        "rmse": float(price["rmse"]),
        "mse": float(price["mse"]),
        "pearson": float(price["pearson"]),
        "spearman": float(price["spearman"]),
        "implied_delta_pearson": float(movement["pearson"]),
        "implied_delta_spearman": float(movement["spearman"]),
        "full_path_mae": float(diagnostics["accepted_unit_full_path_mae"]),
        "timestamp_rank_ic_mean": (
            float(np.mean(timestamp_rank_ic)) if timestamp_rank_ic else float("nan")
        ),
    }
    return payload, arrays, summary


def _load_replayed_model(
    checkpoint: Mapping[str, Any], device: torch.device
) -> XLSTMMixer:
    if checkpoint.get("model_config") != XLSTMMixerConfig().to_dict():
        raise ValueError("checkpoint model configuration mismatch")
    model = XLSTMMixer(backend=checkpoint["config"]["backend"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actual = _probe_output(model, checkpoint["probe"]["raw_input"])
    expected = checkpoint["probe"]["forecast"].cpu()
    if not torch.allclose(actual, expected, rtol=PROBE_RTOL, atol=PROBE_ATOL):
        raise ValueError("same-CUDA-backend fixed-probe replay mismatch")
    return model


def _evaluate_checkpoint(
    checkpoint_path: Path,
    data: XLSTMMixerDataset,
    config: Phase69XLSTMMixerConfig,
    device: torch.device,
    elapsed_training_seconds: float,
    peak_cuda_memory_bytes: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = _load_replayed_model(checkpoint, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if any(
        torch.is_tensor(value) and not torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
    ):
        raise ValueError("checkpoint optimizer state contains non-finite values")
    started = time.perf_counter()
    prediction = _predict_full_path(
        model, data.evaluation_contexts, config.batch_size, device
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    payload, arrays, summary = evaluate_xlstm_predictions(prediction, data)
    snapshot = checkpoint_path.parent
    prediction_path = snapshot / "predictions.npz"
    _atomic_savez(prediction_path, arrays)
    row = {
        "phase": "6.9",
        "method": METHOD,
        "method_id": METHOD_ID,
        "walk": config.walk,
        "epoch": int(checkpoint["completed_epoch"]),
        "seed": config.seed,
        "train_loss": float(checkpoint["history"][-1]["train_loss"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_training_seconds": elapsed_training_seconds,
        "inference_seconds": inference_seconds,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "predictions_sha256": sha256_file(prediction_path),
        **summary,
    }
    write_json(snapshot / "metrics.json", {**row, **payload})
    return row


def run_xlstm_mixer_training(
    dataset_path: Path,
    run_root: Path,
    admission_path: Path,
    config: Phase69XLSTMMixerConfig,
) -> dict[str, Any]:
    """Train or resume one admitted walk-specific XM-MV8 trajectory."""

    dataset_path, run_root, admission_path = map(
        Path, (dataset_path, run_root, admission_path)
    )
    data = load_xlstm_mixer_data(dataset_path, config.walk)
    validate_runtime_admission(admission_path, dataset_path, config)
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("XM-MV8 execution requires the admitted CUDA runtime")
    complete_path = run_root / "training_complete.json"
    if complete_path.is_file():
        raise FileExistsError(f"refusing to overwrite completed XM-MV8 run: {run_root}")

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    if not run_root.exists():
        set_seed(config.seed)
        model = XLSTMMixer(backend=config.backend).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        run_root.mkdir(parents=True)
        environment = json.loads(json.dumps(environment_manifest(device)))
        dependency = xlstm_dependency_manifest()
        write_json(run_root / "config.json", config.to_dict())
        write_json(run_root / "model_config.json", model.config.to_dict())
        write_json(run_root / "environment.json", environment)
        write_json(run_root / "dependency_manifest.json", dependency)
        write_json(
            run_root / "dataset_manifest.json",
            {
                "dataset_path": str(dataset_path.resolve()),
                "dataset_sha256": data.dataset_sha256,
                "manifest_path": str(_data_manifest_path(dataset_path).resolve()),
                "manifest_sha256": data.manifest_sha256,
                "train_identity_sha256": data.train_identity_sha256,
                "evaluation_identity_sha256": data.evaluation_identity_sha256,
                "train_rows": len(data.train_contexts),
                "evaluation_rows": len(data.evaluation_contexts),
                "evaluation_used_for_fitting_or_selection": False,
            },
        )
        write_json(
            run_root / "architecture_manifest.json",
            {
                "method": METHOD,
                "method_id": METHOD_ID,
                "model_config": model.config.to_dict(),
                "source_contract": SOURCE_CONTRACT.to_dict(),
                "source_contract_sha256": SOURCE_CONTRACT.sha256,
                "implementation_sha256": implementation_fingerprint(),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "backend": config.backend,
            },
        )
        write_json(
            run_root / "training_contract.json",
            {
                "optimizer": "Adam",
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "scheduler": None,
                "physical_batch_size": config.batch_size,
                "drop_last": False,
                "objective": "unweighted mean L1 over the accepted-unit [8,5] path",
                "gradient_clip_norm": config.gradient_clip_norm,
                "validation_split": False,
                "early_stopping": False,
                "evaluation_used_for_selection": False,
                "snapshots": list(config.snapshot_epochs),
                "principal_epoch": 50,
            },
        )
        start_epoch = 1
        history: list[dict[str, float]] = []
        raw_probe = torch.from_numpy(data.train_contexts[:4].copy())
        prior_peak_memory = 0
        _atomic_torch_save(
            run_root / "resume.pth",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                config=config,
                data=data,
                admission_path=admission_path,
                completed_epoch=0,
                history=history,
                generator=generator,
                raw_probe=raw_probe,
                environment=environment,
            ),
        )
    else:
        resume_path = run_root / "resume.pth"
        if not resume_path.is_file():
            raise ValueError("incomplete XM-MV8 run has no resume checkpoint")
        resume = torch.load(resume_path, map_location=device, weights_only=True)
        if _config_from_payload(resume["config"], config.device) != config:
            raise ValueError("XM-MV8 resume configuration drift")
        invariants = {
            "dataset_sha256": data.dataset_sha256,
            "data_manifest_sha256": data.manifest_sha256,
            "train_identity_sha256": data.train_identity_sha256,
            "evaluation_identity_sha256": data.evaluation_identity_sha256,
            "runtime_admission_sha256": sha256_file(admission_path),
            "implementation_sha256": implementation_fingerprint(),
            "source_contract_sha256": SOURCE_CONTRACT.sha256,
        }
        if any(resume.get(key) != value for key, value in invariants.items()):
            raise ValueError("XM-MV8 resume provenance drift")
        model = XLSTMMixer(backend=config.backend).to(device)
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        _restore_rng_state(resume["rng_states"], generator)
        start_epoch = int(resume["completed_epoch"]) + 1
        history = [dict(row) for row in resume["history"]]
        raw_probe = resume["probe"]["raw_input"].cpu()
        prior_peak_memory = int(resume.get("peak_cuda_memory_bytes", 0))
        environment = json.loads((run_root / "environment.json").read_text(encoding="utf-8"))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        batches, final_batch_size = shuffled_batches(
            len(data.train_contexts), config.batch_size, generator
        )
        loss_total = 0.0
        rows_seen = 0
        maximum_gradient_norm = 0.0
        for indices in batches:
            index = indices.numpy()
            contexts = torch.from_numpy(data.train_contexts[index]).to(device)
            targets = torch.from_numpy(data.train_targets[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(contexts)
            loss = model.full_path_l1_loss(prediction, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite XM-MV8 loss at epoch {epoch}")
            loss.backward()
            _assert_finite_gradients(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite XM-MV8 gradient norm at epoch {epoch}")
            optimizer.step()
            loss_total += float(loss.detach()) * len(contexts)
            rows_seen += len(contexts)
            maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "train_loss": loss_total / rows_seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "rows_seen": float(rows_seen),
            "batch_count": float(len(batches)),
            "final_batch_size": float(final_batch_size),
            "maximum_gradient_norm_before_clip": maximum_gradient_norm,
            "epoch_seconds": time.perf_counter() - started,
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError("non-finite XM-MV8 epoch diagnostic")
        history.append(row)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            data=data,
            admission_path=admission_path,
            completed_epoch=epoch,
            history=history,
            generator=generator,
            raw_probe=raw_probe,
            environment=environment,
        )
        print(
            f"walk={config.walk} method={METHOD} epoch={epoch} "
            f"loss={row['train_loss']:.8f} batches={len(batches)} "
            f"final_batch={final_batch_size} seconds={row['epoch_seconds']:.2f}",
            flush=True,
        )
        if epoch in config.snapshot_epochs:
            snapshot = run_root / f"e{epoch}"
            snapshot.mkdir(exist_ok=True)
            _atomic_torch_save(snapshot / "checkpoint.pth", payload)
            _atomic_savez(snapshot / "history.npz", _history_arrays(history))
        _atomic_torch_save(run_root / "resume.pth", payload)

    peak_memory = max(prior_peak_memory, int(torch.cuda.max_memory_allocated(device)))
    snapshots: list[dict[str, Any]] = []
    elapsed = [float(row["epoch_seconds"]) for row in history]
    for epoch in config.snapshot_epochs:
        checkpoint_path = run_root / f"e{epoch}" / "checkpoint.pth"
        if not checkpoint_path.is_file():
            raise ValueError(f"XM-MV8 e{epoch} checkpoint is missing")
        snapshots.append(
            _evaluate_checkpoint(
                checkpoint_path,
                data,
                config,
                device,
                elapsed_training_seconds=float(sum(elapsed[:epoch])),
                peak_cuda_memory_bytes=peak_memory,
            )
        )
    write_json(run_root / "sweep_metrics.json", {"snapshots": snapshots})
    result = validate_xlstm_mixer_training(
        dataset_path, run_root, admission_path, device=config.device,
        require_complete=False,
    )
    write_json(
        complete_path,
        {
            "complete": True,
            "phase": "6.9",
            "method": METHOD,
            "walk": config.walk,
            "snapshots": list(config.snapshot_epochs),
            "principal_epoch": 50,
            "selection_rule": "epoch 50 predeclared; no validation or evaluation selection",
        },
    )
    return result


def _metric_close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if math.isnan(float(actual)) and math.isnan(float(expected)):
            return True
        return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-8)
    return actual == expected


def validate_xlstm_mixer_training(
    dataset_path: Path,
    run_root: Path,
    admission_path: Path,
    *,
    device: str = "cuda",
    require_complete: bool = True,
) -> dict[str, Any]:
    """Standalone same-backend checkpoint, prediction, and metric replay."""

    dataset_path, run_root, admission_path = map(
        Path, (dataset_path, run_root, admission_path)
    )
    required = {
        "config.json",
        "model_config.json",
        "environment.json",
        "dependency_manifest.json",
        "dataset_manifest.json",
        "architecture_manifest.json",
        "training_contract.json",
        "sweep_metrics.json",
    }
    if require_complete:
        required.add("training_complete.json")
    missing = sorted(name for name in required if not (run_root / name).is_file())
    if missing:
        raise ValueError(f"XM-MV8 run is missing artifacts: {missing}")
    config = _config_from_payload(
        json.loads((run_root / "config.json").read_text(encoding="utf-8")), device
    )
    data = load_xlstm_mixer_data(dataset_path, config.walk)
    validate_runtime_admission(admission_path, dataset_path, config)
    saved_dependency = json.loads((run_root / "dependency_manifest.json").read_text())
    if _dependency_signature(saved_dependency) != _dependency_signature(xlstm_dependency_manifest()):
        raise ValueError("XM-MV8 saved dependency provenance mismatch")
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("XM-MV8 validation requires its same CUDA backend")
    saved_data = json.loads((run_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    invariants = {
        "dataset_sha256": data.dataset_sha256,
        "manifest_sha256": data.manifest_sha256,
        "train_identity_sha256": data.train_identity_sha256,
        "evaluation_identity_sha256": data.evaluation_identity_sha256,
        "evaluation_used_for_fitting_or_selection": False,
    }
    if any(saved_data.get(key) != value for key, value in invariants.items()):
        raise ValueError("XM-MV8 saved data provenance mismatch")
    rows = json.loads((run_root / "sweep_metrics.json").read_text(encoding="utf-8"))["snapshots"]
    if [int(row["epoch"]) for row in rows] != list(SNAPSHOT_EPOCHS):
        raise ValueError("XM-MV8 snapshot matrix is incomplete")

    replay: dict[str, Any] = {"schema_version": REPLAY_SCHEMA_VERSION, "snapshots": {}}
    for row in rows:
        epoch = int(row["epoch"])
        snapshot = run_root / f"e{epoch}"
        checkpoint_path = snapshot / "checkpoint.pth"
        history_path = snapshot / "history.npz"
        prediction_path = snapshot / "predictions.npz"
        metrics_path = snapshot / "metrics.json"
        if not all(path.is_file() for path in (checkpoint_path, history_path, prediction_path, metrics_path)):
            raise ValueError(f"XM-MV8 e{epoch} artifacts are incomplete")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if sha256_file(checkpoint_path) != metrics.get("checkpoint_sha256"):
            raise ValueError(f"XM-MV8 e{epoch} checkpoint hash mismatch")
        if sha256_file(prediction_path) != metrics.get("predictions_sha256"):
            raise ValueError(f"XM-MV8 e{epoch} prediction hash mismatch")
        checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=True)
        expected_checkpoint = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "phase": "6.9",
            "method": METHOD,
            "method_id": METHOD_ID,
            "walk": config.walk,
            "completed_epoch": epoch,
            "dataset_sha256": data.dataset_sha256,
            "data_manifest_sha256": data.manifest_sha256,
            "train_identity_sha256": data.train_identity_sha256,
            "evaluation_identity_sha256": data.evaluation_identity_sha256,
            "runtime_admission_sha256": sha256_file(admission_path),
            "implementation_sha256": implementation_fingerprint(),
            "source_contract_sha256": SOURCE_CONTRACT.sha256,
        }
        if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
            raise ValueError(f"XM-MV8 e{epoch} checkpoint contract mismatch")
        model = _load_replayed_model(checkpoint, resolved)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if any(
            torch.is_tensor(value) and not torch.isfinite(value).all()
            for state in optimizer.state.values()
            for value in state.values()
        ):
            raise ValueError(f"XM-MV8 e{epoch} optimizer state is non-finite")
        prediction = _predict_full_path(
            model, data.evaluation_contexts, config.batch_size, resolved
        )
        with np.load(prediction_path, allow_pickle=False) as stored:
            if not np.allclose(
                prediction,
                stored["prediction_full_path"],
                rtol=PROBE_RTOL,
                atol=PROBE_ATOL,
            ):
                raise ValueError(f"XM-MV8 e{epoch} prediction replay mismatch")
            for name, expected_values in data.evaluation_metadata.items():
                if not np.array_equal(stored[name], expected_values):
                    raise ValueError(f"XM-MV8 e{epoch} identity mismatch: {name}")
        recomputed, _, summary = evaluate_xlstm_predictions(prediction, data)
        for key, value in summary.items():
            if not _metric_close(value, metrics.get(key)):
                raise ValueError(f"XM-MV8 e{epoch} metric replay mismatch: {key}")
        if recomputed["extra_supervision_disclosure"] != metrics.get(
            "extra_supervision_disclosure"
        ):
            raise ValueError(f"XM-MV8 e{epoch} supervision disclosure mismatch")
        with np.load(history_path, allow_pickle=False) as history:
            if len(history["epochs"]) != epoch or int(history["epochs"][-1]) != epoch:
                raise ValueError(f"XM-MV8 e{epoch} history mismatch")
        replay["snapshots"][f"e{epoch}"] = {
            "checkpoint_replay": True,
            "prediction_replay": True,
            "metric_replay": True,
        }

    if require_complete:
        complete = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
        if (complete.get("complete") is not True
            or complete.get("snapshots") != list(SNAPSHOT_EPOCHS)
            or complete.get("principal_epoch") != 50):
            raise ValueError("XM-MV8 completion marker is invalid")
    replay["valid"] = True
    replay["status"] = "pass"
    write_json(run_root / "replay_validation.json", replay)
    principal = rows[-1]
    return {
        "valid": True,
        "status": "pass",
        "method": METHOD,
        "walk": config.walk,
        "snapshots": list(SNAPSHOT_EPOCHS),
        "epoch50_checkpoint_sha256": principal["checkpoint_sha256"],
        "epoch50_predictions_sha256": principal["predictions_sha256"],
    }


def runtime_environment_payload(device: torch.device) -> dict[str, Any]:
    payload = json.loads(json.dumps(environment_manifest(device)))
    payload.update(
        {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "cuda_compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        }
    )
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        payload["cuda_home"] = CUDA_HOME
    except ImportError:
        payload["cuda_home"] = None
    payload["torch_build_configuration"] = torch.__config__.show()
    payload["runtime_package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "xlstm")
    }
    payload["compiler_toolchain"] = {
        "cxx_path": shutil.which("c++"),
        "cxx_version": _command_version(["c++", "--version"]),
        "nvcc_path": shutil.which("nvcc"),
        "nvcc_version": _command_version(["nvcc", "--version"]),
    }
    try:
        einops_version = importlib.metadata.version("einops")
    except importlib.metadata.PackageNotFoundError:
        einops_version = None
    payload["einops"] = {
        "version": einops_version,
        "required_by_project_adapter": False,
        "note": "used by the audited upstream wrapper, not by the minimal project adapter",
    }
    return payload


def _command_version(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
