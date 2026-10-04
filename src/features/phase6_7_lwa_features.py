"""Frozen 384-wide LWA master representation stores for Phase 6.7."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from baselines.lwa import build_lwa_inference_encoder, build_lwa_joint, build_lwa_mapper_stage
from data_processing.phase5_walks import sha256_arrays, sha256_file, validate_phase5_bundle_files
from data_processing.phase6_volatility_labels import validate_volatility_label_bundle_files
from features.phase5_features import IDENTITY_FIELDS, array_sha256
from training.phase5_encoder import environment_manifest, resolve_device, write_json
from training.phase6_7_lwa import (
    METHOD,
    TRAINING_SCHEMA_VERSION,
    load_lwa_scaler,
    lwa_model_config_from_payload,
    validate_lwa_pretraining,
)


TASKS = ("classification_h2", "absolute_price_h8", "realised_variance")
OUTPUT_DIM = 384
SCHEMA_VERSION = "phase6-7-lwa-master-features-v1"


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _task_validation(task: str, dataset_path: Path) -> dict[str, Any]:
    if task == "realised_variance":
        return validate_volatility_label_bundle_files(dataset_path, replay_source=False)
    return validate_phase5_bundle_files(dataset_path, replay_source=False)


def _load_inference_encoder(
    run_root: Path, *, walk: int, device: torch.device
) -> tuple[torch.nn.Module, Any, Mapping[str, Any]]:
    checkpoint_path = run_root / "mapper" / "e50" / "checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != TRAINING_SCHEMA_VERSION
        or checkpoint.get("method") != METHOD
        or checkpoint.get("stage") != "mapper"
        or int(checkpoint.get("walk", -1)) != walk
        or int(checkpoint.get("completed_epoch", -1)) != 50
    ):
        raise ValueError("only the matching LWA mapper epoch-50 checkpoint may extract features")
    model_config = lwa_model_config_from_payload(checkpoint["model_config"])
    mapper = build_lwa_mapper_stage(build_lwa_joint(model_config))
    mapper.load_state_dict(checkpoint["model_state_dict"], strict=True)
    inference = build_lwa_inference_encoder(mapper).to(device)
    inference.eval()
    scaler = load_lwa_scaler(run_root / "input_scaler.npz")
    return inference, scaler, checkpoint


@torch.no_grad()
def extract_lwa_embeddings(
    sequences: np.ndarray,
    run_root: Path,
    *,
    walk: int,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (64, 5) or not np.isfinite(values).all():
        raise ValueError("LWA extraction requires finite [N,64,5] sequences")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved = resolve_device(device)
    model, scaler, _ = _load_inference_encoder(Path(run_root), walk=walk, device=resolved)
    outputs = []
    for batch in torch.from_numpy(values).split(batch_size):
        normalized = scaler.transform_tensor(batch.to(resolved))
        outputs.append(model(normalized).cpu().numpy().astype(np.float32, copy=False))
    result = np.concatenate(outputs, axis=0)
    if result.shape != (len(values), OUTPUT_DIM) or not np.isfinite(result).all():
        raise ValueError("invalid LWA feature extraction output")
    return result


def _load_task_inputs(
    task: str, dataset_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validation = _task_validation(task, dataset_path)
    arrays: dict[str, np.ndarray] = {}
    with np.load(dataset_path, allow_pickle=False) as stored:
        for split in ("train", "test"):
            arrays[f"{split}_sequences"] = np.asarray(stored[f"{split}_sequences"], dtype=np.float32)
            for field in IDENTITY_FIELDS:
                arrays[f"{split}_{field}"] = np.asarray(stored[f"{split}_{field}"])
    for split in ("train", "test"):
        values = arrays[f"{split}_sequences"]
        if values.ndim != 3 or values.shape[1:] != (64, 5) or not np.isfinite(values).all():
            raise ValueError(f"invalid LWA {task} {split} sequences")
    return arrays, validation


def build_lwa_feature_store(
    task_datasets: Mapping[str, Path],
    encoder_dataset_path: Path,
    cache_root: Path,
    encoder_run_root: Path,
    output_path: Path,
    *,
    walk: int,
    device: str = "cuda",
    batch_size: int = 1024,
) -> dict[str, Any]:
    if walk not in (1, 2) or set(task_datasets) != set(TASKS):
        raise ValueError("LWA master store requires one walk and all three tasks")
    manifest_path = Path(f"{output_path}.manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite LWA feature store: {output_path}")
    training_validation = validate_lwa_pretraining(
        encoder_dataset_path, cache_root, encoder_run_root
    )
    if int(training_validation["walk"]) != walk:
        raise ValueError("LWA training walk mismatch")
    checkpoint_path = encoder_run_root / "mapper" / "e50" / "checkpoint.pth"
    scaler_path = encoder_run_root / "input_scaler.npz"
    saved_arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for task in TASKS:
        dataset_path = Path(task_datasets[task])
        task_inputs, validation = _load_task_inputs(task, dataset_path)
        split_records = {}
        for split in ("train", "test"):
            sequences = task_inputs[f"{split}_sequences"]
            print(
                f"LWA walk {walk} feature extraction: {task}/{split} "
                f"({len(sequences)} rows)",
                flush=True,
            )
            started = time.perf_counter()
            features = extract_lwa_embeddings(
                sequences, encoder_run_root, walk=walk, device=device, batch_size=batch_size
            )
            prefix = f"{task}_{split}"
            saved_arrays[f"{prefix}_features"] = features
            for field in IDENTITY_FIELDS:
                saved_arrays[f"{prefix}_{field}"] = task_inputs[f"{split}_{field}"]
            split_records[split] = {
                "rows": len(features),
                "feature_hash": array_sha256(features),
                "identity_hash": sha256_arrays(
                    *(task_inputs[f"{split}_{field}"] for field in IDENTITY_FIELDS)
                ),
                "sequence_hash": array_sha256(sequences),
                "extraction_seconds": time.perf_counter() - started,
            }
            print(
                f"LWA walk {walk} feature extraction complete: {task}/{split} "
                f"in {split_records[split]['extraction_seconds']:.2f}s",
                flush=True,
            )
        provenance[task] = {
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
            "validation": validation,
            "splits": split_records,
        }
    _atomic_savez(output_path, saved_arrays)
    resolved = resolve_device(device)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "6.7",
        "method": METHOD,
        "walk": walk,
        "output_dim": OUTPUT_DIM,
        "slice_order": ["time[0:128]", "mapped_fourier[128:256]", "mapped_wavelet[256:384]"],
        "task_order": list(TASKS),
        "task_provenance": provenance,
        "encoder_dataset_path": str(Path(encoder_dataset_path).resolve()),
        "encoder_dataset_sha256": sha256_file(encoder_dataset_path),
        "cache_root": str(Path(cache_root).resolve()),
        "cache_manifest_sha256": training_validation["cache_manifest_sha256"],
        "encoder_run_root": str(Path(encoder_run_root).resolve()),
        "mapper_checkpoint_path": str(checkpoint_path.resolve()),
        "mapper_checkpoint_sha256": sha256_file(checkpoint_path),
        "input_scaler_path": str(scaler_path.resolve()),
        "input_scaler_sha256": sha256_file(scaler_path),
        "source_checkpoint_validation": training_validation,
        "targets_loaded_during_extraction": False,
        "transformed_domain_encoders_used_for_features": False,
        "efficient_inference_path_only": True,
        "environment": environment_manifest(resolved),
    }
    manifest["feature_store_sha256"] = sha256_file(output_path)
    write_json(manifest_path, manifest)
    return validate_lwa_feature_store(output_path, replay=False)


def validate_lwa_feature_store(
    feature_path: Path,
    *,
    replay: bool = False,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Any]:
    feature_path = Path(feature_path)
    manifest_path = Path(f"{feature_path}.manifest.json")
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("LWA feature store and manifest are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("method") != METHOD
        or int(manifest.get("output_dim", -1)) != OUTPUT_DIM
    ):
        raise ValueError("unexpected LWA feature store contract")
    if manifest.get("targets_loaded_during_extraction") is not False:
        raise ValueError("LWA target-independence invariant failed")
    if manifest.get("efficient_inference_path_only") is not True:
        raise ValueError("LWA efficient inference boundary is missing")
    if sha256_file(feature_path) != manifest["feature_store_sha256"]:
        raise ValueError("LWA feature store hash mismatch")
    checkpoint_path = Path(manifest["mapper_checkpoint_path"])
    scaler_path = Path(manifest["input_scaler_path"])
    if sha256_file(checkpoint_path) != manifest["mapper_checkpoint_sha256"]:
        raise ValueError("LWA mapper checkpoint hash mismatch")
    if sha256_file(scaler_path) != manifest["input_scaler_sha256"]:
        raise ValueError("LWA input scaler hash mismatch")
    validate_lwa_pretraining(
        Path(manifest["encoder_dataset_path"]),
        Path(manifest["cache_root"]),
        Path(manifest["encoder_run_root"]),
    )
    with np.load(feature_path, allow_pickle=False) as stored:
        expected = {
            f"{task}_{split}_{suffix}"
            for task in TASKS
            for split in ("train", "test")
            for suffix in ("features", *IDENTITY_FIELDS)
        }
        if set(stored.files) != expected:
            raise ValueError("LWA feature store has missing or unexpected arrays")
        for task in TASKS:
            record = manifest["task_provenance"][task]
            dataset_path = Path(record["dataset_path"])
            if sha256_file(dataset_path) != record["dataset_sha256"]:
                raise ValueError(f"LWA feature dataset hash mismatch: {task}")
            _task_validation(task, dataset_path)
            with np.load(dataset_path, allow_pickle=False) as source:
                for split in ("train", "test"):
                    prefix = f"{task}_{split}"
                    features = np.asarray(stored[f"{prefix}_features"])
                    if features.shape != (int(record["splits"][split]["rows"]), OUTPUT_DIM):
                        raise ValueError(f"invalid LWA feature shape: {prefix}")
                    if features.dtype != np.float32 or not np.isfinite(features).all():
                        raise ValueError(f"invalid LWA feature values: {prefix}")
                    if array_sha256(features) != record["splits"][split]["feature_hash"]:
                        raise ValueError(f"LWA feature hash mismatch: {prefix}")
                    for field in IDENTITY_FIELDS:
                        if not np.array_equal(stored[f"{prefix}_{field}"], source[f"{split}_{field}"]):
                            raise ValueError(f"LWA feature identity mismatch: {prefix}_{field}")
                    source_sequences = np.asarray(source[f"{split}_sequences"], dtype=np.float32)
                    if array_sha256(source_sequences) != record["splits"][split]["sequence_hash"]:
                        raise ValueError(f"LWA source sequence hash mismatch: {prefix}")
                    if replay:
                        repeated = extract_lwa_embeddings(
                            source_sequences,
                            Path(manifest["encoder_run_root"]),
                            walk=int(manifest["walk"]),
                            device=device,
                            batch_size=batch_size,
                        )
                        if not np.allclose(repeated, features, rtol=5e-4, atol=2e-5):
                            raise ValueError(f"LWA extraction replay mismatch: {prefix}")
    return {
        "valid": True,
        "method": METHOD,
        "walk": int(manifest["walk"]),
        "output_dim": OUTPUT_DIM,
        "tasks": list(TASKS),
        "replayed": replay,
        "target_independent": True,
    }


def load_lwa_task_features(
    feature_path: Path, task: str, *, validate: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if validate:
        validate_lwa_feature_store(feature_path, replay=False)
    with np.load(feature_path, allow_pickle=False) as stored:
        return (
            np.asarray(stored[f"{task}_train_features"], dtype=np.float32),
            np.asarray(stored[f"{task}_test_features"], dtype=np.float32),
        )
