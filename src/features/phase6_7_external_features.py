"""Frozen Phase 6.7 external-representation master stores."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from baselines.saurl_ts import SaURLInputScaler
from data_processing.phase5_walks import sha256_arrays, sha256_file, validate_phase5_bundle_files
from data_processing.phase6_volatility_labels import validate_volatility_label_bundle_files
from features.phase5_features import IDENTITY_FIELDS, array_sha256
from training.phase5_encoder import environment_manifest, resolve_device, write_json
from training.phase6_7_external_encoders import (
    Phase67ExternalEncoderConfig,
    _config_from_payload,
    _model_config_from_payload,
    build_external_encoder,
    cross_device_tensors_close,
    validate_external_encoder,
)


METHOD = "saurl_frozen"
TASKS = ("classification_h2", "absolute_price_h8", "realised_variance")
OUTPUT_DIM = 128
SCHEMA_VERSION = "phase6-7-external-master-features-v1"


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _task_validation(task: str, dataset_path: Path) -> dict[str, Any]:
    if task == "realised_variance":
        return validate_volatility_label_bundle_files(dataset_path, replay_source=False)
    return validate_phase5_bundle_files(dataset_path)


def _load_checkpoint(
    checkpoint_path: Path, *, walk: int, device: torch.device
) -> tuple[torch.nn.Module, SaURLInputScaler, Phase67ExternalEncoderConfig, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("method") != METHOD
        or int(checkpoint.get("walk", -1)) != walk
        or int(checkpoint.get("completed_epoch", -1)) != 50
    ):
        raise ValueError("only the matching epoch-50 SaURL checkpoint may extract features")
    config = _config_from_payload(checkpoint["config"], str(device))
    model_config = _model_config_from_payload(checkpoint["model_config"])
    model = build_external_encoder(config, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    scaler = SaURLInputScaler.from_state_dict(checkpoint["input_scaler"])
    return model, scaler, config, checkpoint


@torch.no_grad()
def extract_saurl_embeddings(
    sequences: np.ndarray,
    checkpoint_path: Path,
    *,
    walk: int,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    """Encode model-input contexts only; this function has no target interface."""

    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (64, 5) or not np.isfinite(values).all():
        raise ValueError("SaURL extraction requires finite [N,64,5] sequences")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model, scaler, _, _ = _load_checkpoint(checkpoint_path, walk=walk, device=resolved)
    outputs = []
    tensor = torch.from_numpy(values)
    for batch in tensor.split(batch_size):
        normalized = scaler.transform(batch.to(resolved))
        outputs.append(model.encode(normalized).cpu().numpy().astype(np.float32, copy=False))
    result = np.concatenate(outputs, axis=0)
    if result.shape != (len(values), OUTPUT_DIM) or not np.isfinite(result).all():
        raise ValueError("invalid SaURL feature extraction output")
    return result


def _load_task_inputs(
    task: str, dataset_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validation = _task_validation(task, dataset_path)
    arrays: dict[str, np.ndarray] = {}
    with np.load(dataset_path, allow_pickle=False) as stored:
        for split in ("train", "test"):
            # These are the exact walk-preprocessed model inputs. In particular,
            # volume carries the walk-training-only transform used by encoder rows.
            arrays[f"{split}_sequences"] = np.asarray(
                stored[f"{split}_sequences"], dtype=np.float32
            )
            for field in IDENTITY_FIELDS:
                arrays[f"{split}_{field}"] = np.asarray(stored[f"{split}_{field}"])
    for split in ("train", "test"):
        sequences = arrays[f"{split}_sequences"]
        if sequences.ndim != 3 or sequences.shape[1:] != (64, 5):
            raise ValueError(f"invalid {task} {split} sequence shape")
        if not np.isfinite(sequences).all():
            raise ValueError(f"non-finite {task} {split} sequences")
    return arrays, validation


def build_external_feature_store(
    task_datasets: Mapping[str, Path],
    encoder_dataset_path: Path,
    encoder_run_root: Path,
    output_path: Path,
    *,
    walk: int,
    device: str = "cuda",
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Build one SaURL master store containing all three task populations."""

    if walk not in (1, 2) or set(task_datasets) != set(TASKS):
        raise ValueError("master store requires one valid walk and the declared three-task mapping")
    manifest_path = Path(f"{output_path}.manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite external feature store: {output_path}")
    validation = validate_external_encoder(encoder_dataset_path, encoder_run_root)
    if validation["method"] != METHOD or int(validation["walk"]) != walk:
        raise ValueError("external feature checkpoint method/walk mismatch")
    checkpoint_path = encoder_run_root / "e50" / "checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    scaler_state = SaURLInputScaler.from_state_dict(checkpoint["input_scaler"])
    checkpoint_scaler_hash = _scaler_state_hash(scaler_state)
    scaler_path = encoder_run_root / "input_scaler.npz"
    saved_scaler_hash = sha256_file(scaler_path)
    with np.load(scaler_path, allow_pickle=False) as stored_scaler:
        if not np.array_equal(stored_scaler["mean"], scaler_state.mean.numpy()):
            raise ValueError("checkpoint/input-scaler mean mismatch")
        if not np.array_equal(stored_scaler["scale"], scaler_state.scale.numpy()):
            raise ValueError("checkpoint/input-scaler scale mismatch")
        if float(stored_scaler["minimum_scale"]) != scaler_state.minimum_scale:
            raise ValueError("checkpoint/input-scaler minimum-scale mismatch")

    saved_arrays: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for task in TASKS:
        dataset_path = Path(task_datasets[task])
        task_inputs, task_validation = _load_task_inputs(task, dataset_path)
        split_records: dict[str, Any] = {}
        for split in ("train", "test"):
            sequences = task_inputs[f"{split}_sequences"]
            started = time.perf_counter()
            features = extract_saurl_embeddings(
                sequences,
                checkpoint_path,
                walk=walk,
                device=device,
                batch_size=batch_size,
            )
            elapsed = time.perf_counter() - started
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
                "extraction_seconds": elapsed,
            }
        provenance[task] = {
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
            "validation": task_validation,
            "splits": split_records,
        }

    resolved = resolve_device(device)
    cpu_cuda_probe = _cpu_device_probe(
        task_datasets[TASKS[0]], checkpoint_path, walk=walk, extraction_device=resolved
    )
    manual_sum_probe = _manual_sum_probe(
        task_datasets[TASKS[0]],
        saved_arrays[f"{TASKS[0]}_train_features"],
        checkpoint_path,
        walk=walk,
        extraction_device=resolved,
    )
    _atomic_savez(output_path, saved_arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "6.7",
        "method": METHOD,
        "walk": walk,
        "output_dim": OUTPUT_DIM,
        "task_order": list(TASKS),
        "task_provenance": provenance,
        "encoder_dataset_path": str(encoder_dataset_path.resolve()),
        "encoder_dataset_sha256": sha256_file(encoder_dataset_path),
        "encoder_run_root": str(encoder_run_root.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_scaler_file_sha256": saved_scaler_hash,
        "checkpoint_scaler_state_hash": checkpoint_scaler_hash,
        "source_checkpoint_validation": validation,
        "sequence_array_contract": "train_sequences/test_sequences",
        "targets_loaded_during_extraction": False,
        "projectors_predictors_targets_sada_used_for_features": False,
        "cpu_device_probe": cpu_cuda_probe,
        "manual_attention_sum_probe": manual_sum_probe,
        "environment": environment_manifest(resolved),
    }
    manifest["feature_store_sha256"] = sha256_file(output_path)
    write_json(manifest_path, manifest)
    return validate_external_feature_store(output_path, replay=False)


def _scaler_state_hash(scaler: SaURLInputScaler) -> str:
    state = scaler.state_dict()
    return sha256_arrays(
        state["mean"].numpy(),
        state["scale"].numpy(),
        np.asarray(state["minimum_scale"], dtype=np.float64),
    )


def _cpu_device_probe(
    dataset_path: Path,
    checkpoint_path: Path,
    *,
    walk: int,
    extraction_device: torch.device,
) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=False) as stored:
        probe = np.asarray(stored["train_sequences"][:8], dtype=np.float32)
    cpu = extract_saurl_embeddings(probe, checkpoint_path, walk=walk, device="cpu", batch_size=8)
    other = extract_saurl_embeddings(
        probe, checkpoint_path, walk=walk, device=str(extraction_device), batch_size=8
    )
    maximum_absolute_difference = float(np.max(np.abs(cpu - other)))
    close = cross_device_tensors_close(torch.from_numpy(cpu), torch.from_numpy(other))
    if not close:
        raise ValueError("SaURL CPU/extraction-device probe mismatch")
    return {
        "rows": len(probe),
        "extraction_device": str(extraction_device),
        "cpu_hash": array_sha256(cpu),
        "device_hash": array_sha256(other),
        "maximum_absolute_difference": maximum_absolute_difference,
        "close": close,
    }


@torch.no_grad()
def _manual_sum_probe(
    dataset_path: Path,
    saved_features: np.ndarray,
    checkpoint_path: Path,
    *,
    walk: int,
    extraction_device: torch.device,
) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=False) as stored:
        raw = np.asarray(stored["train_sequences"][:8], dtype=np.float32)
    model, scaler, _, _ = _load_checkpoint(
        checkpoint_path, walk=walk, device=extraction_device
    )
    normalized = scaler.transform(torch.from_numpy(raw).to(extraction_device))
    parts = model.encode_parts(normalized)
    manual = parts.weighted_branches.sum(dim=1)
    expected = torch.from_numpy(np.asarray(saved_features[:8])).to(manual)
    maximum_absolute_difference = float((manual - expected).abs().max().cpu())
    if not cross_device_tensors_close(manual, expected):
        raise ValueError("saved SaURL features differ from the manual RwAM weighted sum")
    return {
        "rows": len(raw),
        "device": str(extraction_device),
        "maximum_absolute_difference": maximum_absolute_difference,
        "matches_saved_features": True,
    }


def validate_external_feature_store(
    feature_path: Path,
    *,
    replay: bool = False,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Any]:
    feature_path = Path(feature_path)
    manifest_path = Path(f"{feature_path}.manifest.json")
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("external feature store and manifest are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected external feature schema")
    if manifest.get("method") != METHOD or int(manifest.get("output_dim", -1)) != OUTPUT_DIM:
        raise ValueError("external feature method/width mismatch")
    if manifest.get("targets_loaded_during_extraction") is not False:
        raise ValueError("external feature target-independence invariant failed")
    if manifest.get("manual_attention_sum_probe", {}).get("matches_saved_features") is not True:
        raise ValueError("external feature manual attention-sum probe is missing")
    if sha256_file(feature_path) != manifest.get("feature_store_sha256"):
        raise ValueError("external feature store hash mismatch")
    checkpoint_path = Path(manifest["checkpoint_path"])
    if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("external feature checkpoint hash mismatch")
    encoder_dataset_path = Path(manifest["encoder_dataset_path"])
    if sha256_file(encoder_dataset_path) != manifest["encoder_dataset_sha256"]:
        raise ValueError("external feature encoder-dataset hash mismatch")
    encoder_run_root = Path(manifest["encoder_run_root"])
    validate_external_encoder(encoder_dataset_path, encoder_run_root)
    scaler_path = encoder_run_root / "input_scaler.npz"
    if sha256_file(scaler_path) != manifest["input_scaler_file_sha256"]:
        raise ValueError("external feature input-scaler file hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_scaler = SaURLInputScaler.from_state_dict(checkpoint["input_scaler"])
    if _scaler_state_hash(checkpoint_scaler) != manifest["checkpoint_scaler_state_hash"]:
        raise ValueError("external feature checkpoint scaler-state hash mismatch")

    with np.load(feature_path, allow_pickle=False) as stored:
        expected = {
            f"{task}_{split}_{suffix}"
            for task in TASKS
            for split in ("train", "test")
            for suffix in ("features", *IDENTITY_FIELDS)
        }
        if set(stored.files) != expected:
            raise ValueError("external feature store has missing or unexpected arrays")
        for task in TASKS:
            record = manifest["task_provenance"][task]
            dataset_path = Path(record["dataset_path"])
            if sha256_file(dataset_path) != record["dataset_sha256"]:
                raise ValueError(f"external feature dataset hash mismatch: {task}")
            if (
                sha256_file(Path(f"{dataset_path}.manifest.json"))
                != record["dataset_manifest_sha256"]
            ):
                raise ValueError(f"external feature dataset-manifest hash mismatch: {task}")
            _task_validation(task, dataset_path)
            with np.load(dataset_path, allow_pickle=False) as source:
                for split in ("train", "test"):
                    prefix = f"{task}_{split}"
                    features = np.asarray(stored[f"{prefix}_features"])
                    expected_rows = int(record["splits"][split]["rows"])
                    if features.shape != (expected_rows, OUTPUT_DIM) or features.dtype != np.float32:
                        raise ValueError(f"invalid external feature array: {prefix}")
                    if not np.isfinite(features).all():
                        raise ValueError(f"non-finite external feature array: {prefix}")
                    if array_sha256(features) != record["splits"][split]["feature_hash"]:
                        raise ValueError(f"external feature hash mismatch: {prefix}")
                    for field in IDENTITY_FIELDS:
                        key = f"{prefix}_{field}"
                        if not np.array_equal(stored[key], source[f"{split}_{field}"]):
                            raise ValueError(f"external feature identity mismatch: {key}")
                    identity_hash = sha256_arrays(
                        *(np.asarray(source[f"{split}_{field}"]) for field in IDENTITY_FIELDS)
                    )
                    if identity_hash != record["splits"][split]["identity_hash"]:
                        raise ValueError(f"external feature identity hash mismatch: {prefix}")
                    source_sequences = np.asarray(source[f"{split}_sequences"], dtype=np.float32)
                    if array_sha256(source_sequences) != record["splits"][split]["sequence_hash"]:
                        raise ValueError(f"external feature source-sequence hash mismatch: {prefix}")
                    if replay:
                        repeated = extract_saurl_embeddings(
                            source_sequences,
                            checkpoint_path,
                            walk=int(manifest["walk"]),
                            device=device,
                            batch_size=batch_size,
                        )
                        if device == manifest["environment"]["device"]:
                            if array_sha256(repeated) != record["splits"][split]["feature_hash"]:
                                raise ValueError(f"same-device extraction hash mismatch: {prefix}")
                        elif not cross_device_tensors_close(
                            torch.from_numpy(repeated), torch.from_numpy(features)
                        ):
                            raise ValueError(f"cross-device extraction mismatch: {prefix}")
    return {
        "valid": True,
        "method": METHOD,
        "walk": int(manifest["walk"]),
        "output_dim": OUTPUT_DIM,
        "tasks": list(TASKS),
        "replayed": replay,
        "target_independent": True,
    }


def load_external_task_features(
    feature_path: Path, task: str
) -> tuple[np.ndarray, np.ndarray]:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    validate_external_feature_store(Path(feature_path), replay=False)
    with np.load(feature_path, allow_pickle=False) as stored:
        train = np.asarray(stored[f"{task}_train_features"], dtype=np.float32)
        test = np.asarray(stored[f"{task}_test_features"], dtype=np.float32)
    return train, test
