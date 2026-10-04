"""LWA-specific cached views and two-stage Phase 6.7 pretraining.

The implementation is intentionally separate from the completed SaURL
trajectory.  It owns the expensive transformed-view cache, the joint stage,
and the frozen-encoder representation-mapper stage.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from baselines.lwa import (
    LWAConfig,
    MorletCWT,
    build_lwa_inference_encoder,
    build_lwa_joint,
    build_lwa_mapper_stage,
    orthonormal_rfft,
)
from data_processing.phase5_walks import (
    sha256_arrays,
    sha256_file,
    validate_phase5_bundle_files,
)
from features.phase5_features import array_sha256
from training.phase5_encoder import environment_manifest, resolve_device, set_seed, write_json


METHOD = "lwa_frozen"
REPORTING_LABEL = "LWA-Frozen (paper-guided independent implementation)"
SNAPSHOT_EPOCHS = (5, 15, 50)
CACHE_SCHEMA_VERSION = "phase6-7-lwa-view-cache-v1"
TRAINING_SCHEMA_VERSION = "phase6-7-lwa-pretraining-v1"


@dataclass(frozen=True)
class LWAInputScaler:
    mean: np.ndarray
    scale: np.ndarray
    raw_std: np.ndarray
    minimum_scale: float = 1e-8

    @classmethod
    def fit(cls, sequences: np.ndarray, minimum_scale: float = 1e-8) -> "LWAInputScaler":
        values = np.asarray(sequences, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (64, 5) or not len(values):
            raise ValueError("LWA scaler requires non-empty [N,64,5] sequences")
        if not np.isfinite(values).all() or minimum_scale <= 0.0:
            raise ValueError("LWA scaler requires finite values and a positive threshold")
        mean = values.mean(axis=(0, 1))
        raw_std = values.std(axis=(0, 1), ddof=0)
        return cls(
            mean=mean.astype(np.float32),
            scale=np.where(raw_std < minimum_scale, 1.0, raw_std).astype(np.float32),
            raw_std=raw_std.astype(np.float32),
            minimum_scale=float(minimum_scale),
        )

    def transform_numpy(self, sequences: np.ndarray) -> np.ndarray:
        values = np.asarray(sequences, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (64, 5):
            raise ValueError("LWA scaling requires [N,64,5] sequences")
        result = ((values - self.mean) / self.scale).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise FloatingPointError("LWA scaling produced non-finite values")
        return result

    def transform_tensor(self, sequences: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=sequences.dtype, device=sequences.device)
        scale = torch.as_tensor(self.scale, dtype=sequences.dtype, device=sequences.device)
        result = (sequences - mean) / scale
        if not torch.isfinite(result).all():
            raise FloatingPointError("LWA scaling produced non-finite values")
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "scale": self.scale.copy(),
            "raw_std": self.raw_std.copy(),
            "minimum_scale": self.minimum_scale,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LWAInputScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
            raw_std=np.asarray(state["raw_std"], dtype=np.float32),
            minimum_scale=float(state["minimum_scale"]),
        )


@dataclass(frozen=True)
class Phase67LWATrainingConfig:
    walk: int
    seed: int = 0
    physical_batch_size: int = 128
    joint_epochs: int = 50
    mapper_epochs: int = 50
    snapshot_epochs: tuple[int, ...] = SNAPSHOT_EPOCHS
    learning_rate: float = 3e-3
    weight_decay: float = 1e-6
    device: str = "cuda"

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_epochs", tuple(self.snapshot_epochs))
        if self.walk not in (1, 2):
            raise ValueError("walk must be 1 or 2")
        if self.seed != 0:
            raise ValueError("LWA primary pretraining is frozen to seed 0")
        if self.physical_batch_size != 128:
            raise ValueError("LWA requires the authoritative physical batch size 128")
        if self.joint_epochs != 50 or self.mapper_epochs != 50:
            raise ValueError("LWA requires 50 joint plus 50 mapper epochs")
        if self.snapshot_epochs != SNAPSHOT_EPOCHS:
            raise ValueError("LWA requires 5/15/50 snapshots in each stage")
        if self.learning_rate != 3e-3 or self.weight_decay != 1e-6:
            raise ValueError("LWA optimizer settings differ from the approved contract")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_epochs"] = list(self.snapshot_epochs)
        return payload


def _load_encoder_population(dataset_path: Path, walk: int) -> tuple[np.ndarray, dict[str, Any]]:
    validation = validate_phase5_bundle_files(dataset_path, replay_source=False)
    if int(validation["walk"]) != walk:
        raise ValueError("LWA encoder dataset walk mismatch")
    manifest_path = Path(f"{dataset_path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(dataset_path, allow_pickle=False) as stored:
        sequences = np.asarray(stored["encoder_train_sequences"], dtype=np.float32)
        identities = tuple(
            np.asarray(stored[f"encoder_train_{field}"])
            for field in (
                "condition_ids",
                "window_start_ns",
                "decision_date_ns",
                "decision_availability_ns",
            )
        )
    if sequences.ndim != 3 or sequences.shape[1:] != (64, 5):
        raise ValueError("encoder_train_sequences must have shape [N,64,5]")
    if not np.isfinite(sequences).all():
        raise ValueError("encoder_train_sequences must be finite")
    identity_hash = sha256_arrays(*identities)
    if manifest.get("identity_hashes", {}).get("encoder_train") != identity_hash:
        raise ValueError("encoder training identity hash mismatch")
    return sequences, {"bundle": manifest, "identity_hash": identity_hash}


def _save_scaler(path: Path, scaler: LWAInputScaler) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **scaler.state_dict())
    os.replace(temporary, path)


def load_lwa_scaler(path: Path) -> LWAInputScaler:
    with np.load(path, allow_pickle=False) as stored:
        return LWAInputScaler.from_state_dict({name: stored[name] for name in stored.files})


def _pywavelets_version() -> str:
    import pywt

    return str(pywt.__version__)


def build_lwa_view_cache(
    dataset_path: Path,
    cache_root: Path,
    *,
    walk: int,
    model_config: LWAConfig | None = None,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Build an atomic, disk-backed cache from the target-free encoder rows."""

    dataset_path, cache_root = Path(dataset_path), Path(cache_root)
    config = model_config or LWAConfig()
    if chunk_size <= 0:
        raise ValueError("cache chunk size must be positive")
    if cache_root.exists():
        return validate_lwa_view_cache(dataset_path, cache_root, walk=walk)
    sequences, source = _load_encoder_population(dataset_path, walk)
    scaler = LWAInputScaler.fit(sequences)
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    work = cache_root.with_name(f".{cache_root.name}.work")
    if work.exists():
        raise FileExistsError(f"unfinished LWA cache work directory exists: {work}")
    work.mkdir()
    try:
        row_count = len(sequences)
        time_cache = np.lib.format.open_memmap(
            work / "time.npy", mode="w+", dtype=np.float32, shape=(row_count, 64, 5)
        )
        fourier_cache = np.lib.format.open_memmap(
            work / "fourier.npy", mode="w+", dtype=np.complex64, shape=(row_count, 5, 33)
        )
        wavelet_cache = np.lib.format.open_memmap(
            work / "wavelet.npy", mode="w+", dtype=np.float32, shape=(row_count, 5, 48, 64)
        )
        cwt = MorletCWT(config)
        started = time.perf_counter()
        report_every = max(
            chunk_size,
            ((max(1, row_count // 20) + chunk_size - 1) // chunk_size) * chunk_size,
        )
        for start in range(0, row_count, chunk_size):
            stop = min(start + chunk_size, row_count)
            normalized = scaler.transform_numpy(sequences[start:stop])
            tensor = torch.from_numpy(normalized)
            time_cache[start:stop] = normalized
            fourier_cache[start:stop] = orthonormal_rfft(tensor, config).numpy().astype(
                np.complex64, copy=False
            )
            wavelet_cache[start:stop] = cwt(tensor).numpy()
            if start == 0 or stop == row_count or stop % report_every == 0:
                print(
                    f"LWA walk {walk} cache: {stop}/{row_count} rows "
                    f"({100.0 * stop / row_count:.1f}%)",
                    flush=True,
                )
        for array in (time_cache, fourier_cache, wavelet_cache):
            array.flush()
        del time_cache, fourier_cache, wavelet_cache
        _save_scaler(work / "input_scaler.npz", scaler)
        files = {
            name: {"sha256": sha256_file(work / name), "bytes": (work / name).stat().st_size}
            for name in ("time.npy", "fourier.npy", "wavelet.npy", "input_scaler.npz")
        }
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "phase": "6.7",
            "method": METHOD,
            "walk": walk,
            "row_count": row_count,
            "chunk_size": chunk_size,
            "source_dataset_path": str(dataset_path.resolve()),
            "source_dataset_sha256": sha256_file(dataset_path),
            "source_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
            "encoder_train_identity_hash": source["identity_hash"],
            "encoder_train_sequence_hash": array_sha256(sequences),
            "scaler_fit_population": "encoder_train_sequences across rows and time only",
            "scaler_evaluation_rows_used": False,
            "scaler_state_hash": sha256_arrays(
                scaler.mean, scaler.scale, scaler.raw_std, np.asarray(scaler.minimum_scale)
            ),
            "arrays": {
                "time": {"file": "time.npy", "shape": [row_count, 64, 5], "dtype": "float32"},
                "fourier": {"file": "fourier.npy", "shape": [row_count, 5, 33], "dtype": "complex64"},
                "wavelet": {"file": "wavelet.npy", "shape": [row_count, 5, 48, 64], "dtype": "float32"},
            },
            "transform_contract": {
                "fourier": "torch.fft.rfft over time, norm=ortho",
                "wavelet": "PyWavelets channelwise cmor1-1 magnitude",
                "wavelet_scales": list(config.wavelet_scales),
                "pywavelets_version": _pywavelets_version(),
            },
            "model_config": config.to_dict(),
            "files": files,
            "build_seconds": time.perf_counter() - started,
        }
        write_json(work / "manifest.json", manifest)
        os.replace(work, cache_root)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return validate_lwa_view_cache(dataset_path, cache_root, walk=walk)


def validate_lwa_view_cache(
    dataset_path: Path, cache_root: Path, *, walk: int
) -> dict[str, Any]:
    dataset_path, cache_root = Path(dataset_path), Path(cache_root)
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("LWA cache manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unexpected LWA cache schema")
    if manifest.get("method") != METHOD or int(manifest.get("walk", -1)) != walk:
        raise ValueError("LWA cache method/walk mismatch")
    if manifest.get("model_config") != LWAConfig().to_dict():
        raise ValueError("LWA cache model/transform configuration mismatch")
    if manifest.get("transform_contract", {}).get("pywavelets_version") != _pywavelets_version():
        raise ValueError("LWA cache PyWavelets runtime mismatch")
    sequences, source = _load_encoder_population(dataset_path, walk)
    if sha256_file(dataset_path) != manifest["source_dataset_sha256"]:
        raise ValueError("LWA cache source dataset hash mismatch")
    if sha256_file(Path(f"{dataset_path}.manifest.json")) != manifest["source_manifest_sha256"]:
        raise ValueError("LWA cache source manifest hash mismatch")
    if source["identity_hash"] != manifest["encoder_train_identity_hash"]:
        raise ValueError("LWA cache row identity mismatch")
    if array_sha256(sequences) != manifest["encoder_train_sequence_hash"]:
        raise ValueError("LWA cache source sequence mismatch")
    expected = {
        "time": ((len(sequences), 64, 5), np.dtype(np.float32)),
        "fourier": ((len(sequences), 5, 33), np.dtype(np.complex64)),
        "wavelet": ((len(sequences), 5, 48, 64), np.dtype(np.float32)),
    }
    for name, (shape, dtype) in expected.items():
        path = cache_root / manifest["arrays"][name]["file"]
        if sha256_file(path) != manifest["files"][path.name]["sha256"]:
            raise ValueError(f"LWA cache file hash mismatch: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        finite = all(
            np.isfinite(array[start : start + 64]).all()
            for start in range(0, len(array), 64)
        )
        if array.shape != shape or array.dtype != dtype or not finite:
            raise ValueError(f"invalid LWA cached array: {name}")
    scaler_path = cache_root / "input_scaler.npz"
    if sha256_file(scaler_path) != manifest["files"]["input_scaler.npz"]["sha256"]:
        raise ValueError("LWA cache scaler file hash mismatch")
    scaler = load_lwa_scaler(scaler_path)
    state_hash = sha256_arrays(
        scaler.mean, scaler.scale, scaler.raw_std, np.asarray(scaler.minimum_scale)
    )
    if state_hash != manifest["scaler_state_hash"]:
        raise ValueError("LWA cache scaler state mismatch")
    return {
        "valid": True,
        "method": METHOD,
        "walk": walk,
        "rows": len(sequences),
        "physical_batches": len(sequences) // 128,
        "dropped_rows_per_epoch": len(sequences) % 128,
        "cache_manifest_sha256": sha256_file(manifest_path),
        "pywavelets_version": manifest["transform_contract"]["pywavelets_version"],
    }


def drop_last_batch_indices(
    row_count: int, batch_size: int, generator: torch.Generator
) -> list[torch.Tensor]:
    if row_count < batch_size or batch_size != 128:
        raise ValueError("LWA requires at least one physical batch of exactly 128")
    permutation = torch.randperm(row_count, generator=generator)
    usable = row_count - row_count % batch_size
    batches = list(permutation[:usable].split(batch_size))
    if not batches or any(len(batch) != 128 for batch in batches):
        raise AssertionError("LWA drop-last sampler violated the fixed-batch contract")
    return batches


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "name": name,
        "keys": torch.from_numpy(keys.copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _capture_rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
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
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)
    generator.set_state(state["sampler_generator"].cpu())


def _config_from_payload(payload: Mapping[str, Any], device: str) -> Phase67LWATrainingConfig:
    values = dict(payload)
    values["snapshot_epochs"] = tuple(values["snapshot_epochs"])
    values["device"] = device
    return Phase67LWATrainingConfig(**values)


def lwa_model_config_from_payload(payload: Mapping[str, Any]) -> LWAConfig:
    names = {field.name for field in fields(LWAConfig)}
    values = {name: payload[name] for name in names if name in payload}
    if "fourier_branch_channels" in values:
        values["fourier_branch_channels"] = tuple(values["fourier_branch_channels"])
    return LWAConfig(**values)


def _cache_arrays(cache_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.load(cache_root / name, mmap_mode="r", allow_pickle=False)
        for name in ("time.npy", "fourier.npy", "wavelet.npy")
    )  # type: ignore[return-value]


def _batches(
    cache_root: Path, generator: torch.Generator, device: torch.device
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    time_view, fourier_view, wavelet_view = _cache_arrays(cache_root)
    for indices in drop_last_batch_indices(len(time_view), 128, generator):
        index = indices.numpy()
        yield (
            torch.from_numpy(np.asarray(time_view[index])).to(device),
            torch.from_numpy(np.asarray(fourier_view[index])).to(device),
            torch.from_numpy(np.asarray(wavelet_view[index])).to(device),
        )


def _assert_finite_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    gradients = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("LWA update has missing or non-finite gradients")


def _model_probe(model: torch.nn.Module, cache_root: Path, device: torch.device) -> dict[str, Any]:
    time_view, fourier_view, wavelet_view = _cache_arrays(cache_root)
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(np.asarray(time_view[:4])).to(device),
            torch.from_numpy(np.asarray(fourier_view[:4])).to(device),
            torch.from_numpy(np.asarray(wavelet_view[:4])).to(device),
        )
    representations = output.representations
    return {
        "time": representations.time.detach().cpu(),
        "fourier": representations.fourier.detach().cpu(),
        "wavelet": representations.wavelet.detach().cpu(),
    }


def _snapshot_payload(
    *, stage: str, epoch: int, config: Phase67LWATrainingConfig,
    model_config: LWAConfig, model: torch.nn.Module, optimizers: Mapping[str, Any],
    scheduler: Any, histories: Mapping[str, list[float]], rng_state: Mapping[str, Any],
    dataset_hash: str, cache_hash: str, probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "phase": "6.7",
        "method": METHOD,
        "stage": stage,
        "walk": config.walk,
        "completed_epoch": epoch,
        "config": config.to_dict(),
        "model_config": model_config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {name: value.state_dict() for name, value in optimizers.items()},
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "histories": dict(histories),
        "rng_state": dict(rng_state),
        "dataset_sha256": dataset_hash,
        "cache_manifest_sha256": cache_hash,
        "probe": dict(probe),
    }


def _write_history(path: Path, histories: Mapping[str, list[float]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **{name: np.asarray(value, dtype=np.float64) for name, value in histories.items()})
    os.replace(temporary, path)


def _load_resume(path: Path, *, device: torch.device) -> Mapping[str, Any] | None:
    return torch.load(path, map_location=device, weights_only=True) if path.is_file() else None


def run_lwa_pretraining(
    dataset_path: Path,
    cache_root: Path,
    run_root: Path,
    config: Phase67LWATrainingConfig,
    admission_manifest_path: Path,
    model_config: LWAConfig | None = None,
) -> dict[str, Any]:
    """Train or resume one complete 50+50 epoch LWA walk trajectory."""

    dataset_path, cache_root, run_root = Path(dataset_path), Path(cache_root), Path(run_root)
    admission_manifest_path = Path(admission_manifest_path)
    model_config = model_config or LWAConfig()
    admission = json.loads(admission_manifest_path.read_text(encoding="utf-8"))
    if admission.get("method") != METHOD or admission.get("admitted_for_training") is not True:
        raise ValueError("LWA training requires its admitted feasibility manifest")
    admission_hash = sha256_file(admission_manifest_path)
    cache_validation = validate_lwa_view_cache(dataset_path, cache_root, walk=config.walk)
    if (run_root / "training_complete.json").is_file():
        return validate_lwa_pretraining(dataset_path, cache_root, run_root)
    device = resolve_device(config.device)
    set_seed(config.seed)
    sampler = torch.Generator().manual_seed(config.seed)
    dataset_hash = sha256_file(dataset_path)
    cache_hash = cache_validation["cache_manifest_sha256"]
    if not run_root.exists():
        run_root.mkdir(parents=True)
        write_json(run_root / "config.json", config.to_dict())
        write_json(run_root / "model_config.json", model_config.to_dict())
        write_json(run_root / "environment.json", {**environment_manifest(device), "pywavelets": _pywavelets_version()})
        write_json(
            run_root / "admission_manifest.json",
            {
                "path": str(admission_manifest_path.resolve()),
                "sha256": admission_hash,
                "admitted_for_training": True,
            },
        )
        shutil.copyfile(cache_root / "input_scaler.npz", run_root / "input_scaler.npz")
        write_json(
            run_root / "data_manifest.json",
            {
                "dataset_path": str(dataset_path.resolve()),
                "dataset_sha256": dataset_hash,
                "cache_root": str(cache_root.resolve()),
                "cache_manifest_sha256": cache_hash,
                "rows": cache_validation["rows"],
                "physical_batch_size": 128,
                "physical_batches": cache_validation["physical_batches"],
                "dropped_rows_per_epoch": cache_validation["dropped_rows_per_epoch"],
                "drop_last": True,
                "gradient_accumulation": False,
                "evaluation_rows_used": False,
            },
        )
    else:
        saved = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
        if _config_from_payload(saved, config.device).to_dict() != config.to_dict():
            raise ValueError("LWA resume configuration mismatch")
        saved_model = json.loads((run_root / "model_config.json").read_text(encoding="utf-8"))
        if saved_model != model_config.to_dict():
            raise ValueError("LWA resume model configuration mismatch")
        saved_admission = json.loads(
            (run_root / "admission_manifest.json").read_text(encoding="utf-8")
        )
        if saved_admission.get("sha256") != admission_hash:
            raise ValueError("LWA resume feasibility-admission mismatch")
        data_manifest = json.loads((run_root / "data_manifest.json").read_text(encoding="utf-8"))
        if data_manifest["dataset_sha256"] != dataset_hash or data_manifest["cache_manifest_sha256"] != cache_hash:
            raise ValueError("LWA resume data/cache mismatch")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_started = time.perf_counter()

    joint = build_lwa_joint(model_config).to(device)
    joint_optimizer = torch.optim.Adam(
        joint.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        joint_optimizer, T_max=config.joint_epochs, eta_min=0.0
    )
    joint_histories: dict[str, list[float]] = {
        name: [] for name in (
            "total", "time_fourier", "time_wavelet", "fourier_wavelet",
            "mapped_fourier", "mapped_wavelet", "learning_rate", "epoch_seconds",
        )
    }
    joint_resume = _load_resume(run_root / "joint" / "resume.pth", device=device)
    joint_start = 1
    if joint_resume is not None:
        if joint_resume["dataset_sha256"] != dataset_hash or joint_resume["cache_manifest_sha256"] != cache_hash:
            raise ValueError("LWA joint resume provenance mismatch")
        joint.load_state_dict(joint_resume["model_state_dict"], strict=True)
        joint_optimizer.load_state_dict(joint_resume["optimizer_state_dicts"]["joint"])
        scheduler.load_state_dict(joint_resume["scheduler_state_dict"])
        joint_histories = {name: list(values) for name, values in joint_resume["histories"].items()}
        _restore_rng_state(joint_resume["rng_state"], sampler)
        joint_start = int(joint_resume["completed_epoch"]) + 1
    for epoch in range(joint_start, config.joint_epochs + 1):
        started = time.perf_counter()
        totals = {name: 0.0 for name in joint_histories if name not in ("learning_rate", "epoch_seconds")}
        count = 0
        joint.train()
        learning_rate = float(joint_optimizer.param_groups[0]["lr"])
        for views in _batches(cache_root, sampler, device):
            joint_optimizer.zero_grad(set_to_none=True)
            output = joint(*views)
            if not torch.isfinite(output.total_loss):
                raise FloatingPointError("non-finite LWA joint loss")
            output.total_loss.backward()
            _assert_finite_gradients(joint.parameters())
            joint_optimizer.step()
            batch = len(views[0])
            for name in totals:
                value = output.losses.total if name == "total" else getattr(output.losses, name)
                totals[name] += float(value.detach()) * batch
            count += batch
        scheduler.step()
        for name, total in totals.items():
            joint_histories[name].append(total / count)
        joint_histories["learning_rate"].append(learning_rate)
        joint_histories["epoch_seconds"].append(time.perf_counter() - started)
        rng = _capture_rng_state(sampler)
        probe = _model_probe(joint, cache_root, device)
        payload = _snapshot_payload(
            stage="joint", epoch=epoch, config=config, model_config=model_config,
            model=joint, optimizers={"joint": joint_optimizer}, scheduler=scheduler,
            histories=joint_histories, rng_state=rng, dataset_hash=dataset_hash,
            cache_hash=cache_hash, probe=probe,
        )
        _atomic_torch_save(run_root / "joint" / "resume.pth", payload)
        if epoch in SNAPSHOT_EPOCHS:
            snapshot = run_root / "joint" / f"e{epoch}"
            _atomic_torch_save(snapshot / "checkpoint.pth", payload)
            _write_history(snapshot / "history.npz", joint_histories)
        print(
            f"LWA walk {config.walk} joint epoch {epoch}/{config.joint_epochs}: "
            f"loss={joint_histories['total'][-1]:.6f}, "
            f"seconds={joint_histories['epoch_seconds'][-1]:.2f}",
            flush=True,
        )

    joint_final = torch.load(run_root / "joint" / "e50" / "checkpoint.pth", map_location=device, weights_only=True)
    joint.load_state_dict(joint_final["model_state_dict"], strict=True)
    mapper = build_lwa_mapper_stage(joint).to(device)
    fourier_optimizer = torch.optim.Adam(
        mapper.representation_mappers.fourier.parameters(),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    wavelet_optimizer = torch.optim.Adam(
        mapper.representation_mappers.wavelet.parameters(),
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    mapper_histories: dict[str, list[float]] = {
        "total": [], "fourier": [], "wavelet": [], "epoch_seconds": []
    }
    mapper_resume = _load_resume(run_root / "mapper" / "resume.pth", device=device)
    mapper_start = 1
    if mapper_resume is not None:
        if mapper_resume["dataset_sha256"] != dataset_hash or mapper_resume["cache_manifest_sha256"] != cache_hash:
            raise ValueError("LWA mapper resume provenance mismatch")
        mapper.load_state_dict(mapper_resume["model_state_dict"], strict=True)
        fourier_optimizer.load_state_dict(mapper_resume["optimizer_state_dicts"]["fourier"])
        wavelet_optimizer.load_state_dict(mapper_resume["optimizer_state_dicts"]["wavelet"])
        mapper_histories = {name: list(values) for name, values in mapper_resume["histories"].items()}
        _restore_rng_state(mapper_resume["rng_state"], sampler)
        mapper_start = int(mapper_resume["completed_epoch"]) + 1
    for epoch in range(mapper_start, config.mapper_epochs + 1):
        started = time.perf_counter()
        totals = {"total": 0.0, "fourier": 0.0, "wavelet": 0.0}
        count = 0
        mapper.train()
        for views in _batches(cache_root, sampler, device):
            fourier_optimizer.zero_grad(set_to_none=True)
            wavelet_optimizer.zero_grad(set_to_none=True)
            output = mapper(*views)
            if not torch.isfinite(output.total_loss):
                raise FloatingPointError("non-finite LWA mapper loss")
            output.total_loss.backward()
            _assert_finite_gradients(mapper.mapper_parameters())
            fourier_optimizer.step()
            wavelet_optimizer.step()
            batch = len(views[0])
            totals["total"] += float(output.total_loss.detach()) * batch
            totals["fourier"] += float(output.fourier_loss.detach()) * batch
            totals["wavelet"] += float(output.wavelet_loss.detach()) * batch
            count += batch
        for name, total in totals.items():
            mapper_histories[name].append(total / count)
        mapper_histories["epoch_seconds"].append(time.perf_counter() - started)
        rng = _capture_rng_state(sampler)
        probe = _model_probe(mapper, cache_root, device)
        payload = _snapshot_payload(
            stage="mapper", epoch=epoch, config=config, model_config=model_config,
            model=mapper, optimizers={"fourier": fourier_optimizer, "wavelet": wavelet_optimizer},
            scheduler=None, histories=mapper_histories, rng_state=rng,
            dataset_hash=dataset_hash, cache_hash=cache_hash, probe=probe,
        )
        _atomic_torch_save(run_root / "mapper" / "resume.pth", payload)
        if epoch in SNAPSHOT_EPOCHS:
            snapshot = run_root / "mapper" / f"e{epoch}"
            _atomic_torch_save(snapshot / "checkpoint.pth", payload)
            _write_history(snapshot / "history.npz", mapper_histories)
        print(
            f"LWA walk {config.walk} mapper epoch {epoch}/{config.mapper_epochs}: "
            f"loss={mapper_histories['total'][-1]:.6f}, "
            f"seconds={mapper_histories['epoch_seconds'][-1]:.2f}",
            flush=True,
        )

    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    artifacts = {}
    for stage in ("joint", "mapper"):
        artifacts[stage] = {
            f"e{epoch}": {
                "checkpoint_sha256": sha256_file(run_root / stage / f"e{epoch}" / "checkpoint.pth"),
                "history_sha256": sha256_file(run_root / stage / f"e{epoch}" / "history.npz"),
            }
            for epoch in SNAPSHOT_EPOCHS
        }
    write_json(run_root / "artifact_manifest.json", artifacts)
    write_json(
        run_root / "training_complete.json",
        {
            "complete": True,
            "method": METHOD,
            "walk": config.walk,
            "joint_epochs": 50,
            "mapper_epochs": 50,
            "snapshots_per_stage": list(SNAPSHOT_EPOCHS),
            "selection_rule": "joint e50 plus mapper e50 fixed before evaluation",
            "evaluation_used_for_selection": False,
            "total_training_seconds": time.perf_counter() - total_started,
            "peak_cuda_memory_bytes": peak,
            "joint_parameters": sum(p.numel() for p in joint.parameters()),
            "mapper_stage_trainable_parameters": sum(p.numel() for p in mapper.mapper_parameters()),
            "retained_inference_parameters": sum(
                p.numel() for p in build_lwa_inference_encoder(mapper).parameters()
            ),
            "finite_loss_and_gradient_checks_passed": True,
            "rng_and_sampler_state_saved_each_epoch": True,
            "optimizer_states_saved_each_epoch": {
                "joint": ["joint"],
                "mapper": ["fourier", "wavelet"],
            },
        },
    )
    return validate_lwa_pretraining(dataset_path, cache_root, run_root)


def _assert_probe_close(saved: Mapping[str, torch.Tensor], replayed: Mapping[str, Any]) -> None:
    for name in ("time", "fourier", "wavelet"):
        torch.testing.assert_close(saved[name].cpu(), replayed[name].cpu(), rtol=1e-5, atol=1e-6)


def validate_lwa_pretraining(
    dataset_path: Path, cache_root: Path, run_root: Path
) -> dict[str, Any]:
    dataset_path, cache_root, run_root = Path(dataset_path), Path(cache_root), Path(run_root)
    config_payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config = _config_from_payload(config_payload, "cpu")
    model_config = lwa_model_config_from_payload(
        json.loads((run_root / "model_config.json").read_text(encoding="utf-8"))
    )
    cache = validate_lwa_view_cache(dataset_path, cache_root, walk=config.walk)
    admission_record = json.loads(
        (run_root / "admission_manifest.json").read_text(encoding="utf-8")
    )
    admission_path = Path(admission_record["path"])
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if (
        admission.get("admitted_for_training") is not True
        or admission.get("method") != METHOD
        or sha256_file(admission_path) != admission_record["sha256"]
    ):
        raise ValueError("LWA feasibility admission provenance mismatch")
    dataset_hash, cache_hash = sha256_file(dataset_path), cache["cache_manifest_sha256"]
    completed = json.loads((run_root / "training_complete.json").read_text(encoding="utf-8"))
    if completed.get("complete") is not True or completed.get("evaluation_used_for_selection") is not False:
        raise ValueError("invalid LWA completion marker")
    artifacts = json.loads((run_root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(run_root / "input_scaler.npz") != sha256_file(cache_root / "input_scaler.npz"):
        raise ValueError("LWA run/cache scaler mismatch")
    final_models: dict[str, torch.nn.Module] = {}
    for stage in ("joint", "mapper"):
        for epoch in SNAPSHOT_EPOCHS:
            snapshot = run_root / stage / f"e{epoch}"
            checkpoint_path, history_path = snapshot / "checkpoint.pth", snapshot / "history.npz"
            if not checkpoint_path.is_file() or not history_path.is_file():
                raise ValueError(f"missing LWA {stage} e{epoch} snapshot")
            record = artifacts[stage][f"e{epoch}"]
            if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                raise ValueError(f"LWA {stage} e{epoch} checkpoint hash mismatch")
            if sha256_file(history_path) != record["history_sha256"]:
                raise ValueError(f"LWA {stage} e{epoch} history hash mismatch")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("schema_version") != TRAINING_SCHEMA_VERSION
                or checkpoint.get("stage") != stage
                or int(checkpoint.get("completed_epoch", -1)) != epoch
                or checkpoint.get("dataset_sha256") != dataset_hash
                or checkpoint.get("cache_manifest_sha256") != cache_hash
            ):
                raise ValueError(f"invalid LWA {stage} e{epoch} checkpoint contract")
            joint = build_lwa_joint(model_config)
            model = joint if stage == "joint" else build_lwa_mapper_stage(joint)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            replayed = _model_probe(model, cache_root, torch.device("cpu"))
            _assert_probe_close(checkpoint["probe"], replayed)
            with np.load(history_path, allow_pickle=False) as history:
                if any(len(history[name]) != epoch for name in history.files):
                    raise ValueError(f"invalid LWA {stage} e{epoch} history length")
                if any(not np.isfinite(history[name]).all() for name in history.files):
                    raise ValueError(f"non-finite LWA {stage} history")
            if epoch == 50:
                final_models[stage] = model
    mapper = final_models["mapper"]
    joint = final_models["joint"]
    for name in ("time_encoder", "fourier_encoder", "wavelet_encoder"):
        joint_state = getattr(joint, name).state_dict()
        mapper_state = getattr(mapper, name).state_dict()
        if set(joint_state) != set(mapper_state) or any(
            not torch.equal(joint_state[key], mapper_state[key]) for key in joint_state
        ):
            raise ValueError(f"LWA mapper-stage frozen {name} differs from joint e50")
    inference = build_lwa_inference_encoder(mapper)  # type: ignore[arg-type]
    time_view = np.load(cache_root / "time.npy", mmap_mode="r", allow_pickle=False)
    with torch.no_grad():
        features = inference(torch.from_numpy(np.asarray(time_view[:4])))
    if features.shape != (4, 384) or not torch.isfinite(features).all():
        raise ValueError("invalid LWA final inference probe")
    return {
        "valid": True,
        "method": METHOD,
        "walk": config.walk,
        "joint_snapshots": list(SNAPSHOT_EPOCHS),
        "mapper_snapshots": list(SNAPSHOT_EPOCHS),
        "output_dim": 384,
        "cache_manifest_sha256": cache_hash,
    }


def smoke_test_lwa_pretraining() -> dict[str, Any]:
    set_seed(0)
    config = LWAConfig()
    joint = build_lwa_joint(config)
    time_view = torch.randn(2, 64, 5)
    fourier_view = orthonormal_rfft(time_view, config)
    wavelet_view = torch.rand(2, 5, 48, 64)
    output = joint(time_view, fourier_view, wavelet_view)
    output.total_loss.backward()
    _assert_finite_gradients(joint.parameters())
    mapper = build_lwa_mapper_stage(joint)
    mapped = mapper(time_view, fourier_view, wavelet_view)
    mapped.total_loss.backward()
    _assert_finite_gradients(mapper.mapper_parameters())
    inference = build_lwa_inference_encoder(mapper)
    features = inference(time_view)
    return {
        "valid": True,
        "method": METHOD,
        "joint_loss_finite": bool(torch.isfinite(output.total_loss)),
        "mapper_loss_finite": bool(torch.isfinite(mapped.total_loss)),
        "inference_shape": list(features.shape),
    }


def resource_smoke_lwa(cache_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    """Run the one authorized fixed-batch-128 forward/backward resource check."""

    resolved = resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("the authoritative LWA resource smoke requires CUDA")
    set_seed(0)
    torch.cuda.reset_peak_memory_stats(resolved)
    time_view, fourier_view, wavelet_view = _cache_arrays(Path(cache_root))
    if len(time_view) < 128:
        raise ValueError("LWA cache has fewer than 128 rows")
    views = tuple(torch.from_numpy(np.asarray(value[:128])).to(resolved) for value in (time_view, fourier_view, wavelet_view))
    model = build_lwa_joint().to(resolved)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-6)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = model(*views)
    output.total_loss.backward()
    _assert_finite_gradients(model.parameters())
    optimizer.step()
    torch.cuda.synchronize(resolved)
    return {
        "valid": True,
        "physical_batch_size": 128,
        "loss": float(output.total_loss.detach()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(resolved)),
    }
