"""Phase 6.5A deep-LSTM extraction and controlled feature configurations."""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data_processing.phase5_walks import sha256_file
from features.phase5_features import (
    BRANCH_DIMS,
    BRANCH_ORDER,
    IDENTITY_FIELDS,
    array_sha256,
    validate_phase5_feature_bundle,
)
from features.phase6_volatility_features import validate_phase6_volatility_feature_bundle
from training.phase5_encoder import environment_manifest, resolve_device, write_json
from training.phase6_5_lstm_capacity import (
    VARIANTS,
    Phase65LSTMCapacityConfig,
    build_lstm_capacity_model,
    validate_lstm_capacity_encoder,
)


SCHEMA_VERSION = "phase6-5a-lstm-capacity-features-v1"
TASKS = ("classification_h2", "absolute_price_h8", "realised_variance")
MASTER_BRANCH_DIMS = {
    **BRANCH_DIMS,
    "contrastive_lstm2": 128,
    "byol_lstm2": 128,
}
CONFIG_BRANCHES: dict[str, tuple[str, ...]] = {
    "H0": BRANCH_ORDER,
    "HC-SL2": ("statistical", "transformed", "vae", "contrastive_lstm2", "byol"),
    "HB-SL2": ("statistical", "transformed", "vae", "contrastive", "byol_lstm2"),
    "HC-AL2": (*BRANCH_ORDER, "contrastive_lstm2"),
    "HB-AL2": (*BRANCH_ORDER, "byol_lstm2"),
}
CONFIG_DIMS = {
    name: sum(MASTER_BRANCH_DIMS[branch] for branch in branches)
    for name, branches in CONFIG_BRANCHES.items()
}


def _validate_h0_source(task: str, dataset_path: Path, feature_path: Path) -> dict[str, Any]:
    if task == "realised_variance":
        return validate_phase6_volatility_feature_bundle(feature_path)
    return validate_phase5_feature_bundle(feature_path, dataset_path=dataset_path)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _checkpoint_path(encoder_root: Path, walk: int, variant: str) -> Path:
    family = "contrastive" if variant.startswith("contrastive") else "byol"
    return (
        encoder_root
        / "pretraining"
        / f"walk{walk}"
        / family
        / "lstm2"
        / "seed0"
        / "e50"
        / "checkpoint.pth"
    )


@torch.no_grad()
def extract_lstm2_branch(
    sequences: np.ndarray,
    checkpoint_path: Path,
    *,
    walk: int,
    variant: str,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    run_root = checkpoint_path.parents[1]
    config_payload = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    config_payload["snapshot_epochs"] = tuple(config_payload["snapshot_epochs"])
    config = Phase65LSTMCapacityConfig(**{**config_payload, "device": device})
    if config.walk != walk or config.variant != variant:
        raise ValueError("Phase 6.5A checkpoint walk/variant mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("completed_epoch") != 50:
        raise ValueError("only the predeclared epoch-50 Phase 6.5A checkpoint may extract")
    resolved = resolve_device(device)
    model = build_lstm_capacity_model(config).to(resolved)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
    result = np.concatenate(
        [model.encode(batch.to(resolved)).cpu().numpy().astype(np.float32) for batch in tensor.split(batch_size)]
    )
    if result.shape != (len(sequences), 128) or not np.isfinite(result).all():
        raise ValueError("invalid Phase 6.5A feature extraction output")
    return result


def build_lstm_capacity_feature_store(
    dataset_path: Path,
    h0_feature_path: Path,
    encoder_dataset_path: Path,
    encoder_root: Path,
    output_path: Path,
    *,
    task: str,
    walk: int,
    device: str = "cuda",
    batch_size: int = 1024,
) -> dict[str, Any]:
    if task not in TASKS or walk not in (1, 2):
        raise ValueError("invalid Phase 6.5A feature task or walk")
    manifest_path = Path(f"{output_path}.manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase 6.5A feature store: {output_path}")
    for path in (dataset_path, h0_feature_path, encoder_dataset_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    h0_validation = _validate_h0_source(task, dataset_path, h0_feature_path)
    arrays: dict[str, np.ndarray] = {}
    with np.load(dataset_path, allow_pickle=False) as data, np.load(
        h0_feature_path, allow_pickle=False
    ) as h0:
        sequences = {}
        for split in ("train", "test"):
            sequences[split] = np.asarray(data[f"{split}_sequences"], dtype=np.float32)
            for field in IDENTITY_FIELDS:
                key = f"{split}_{field}"
                if not np.array_equal(data[key], h0[key]):
                    raise ValueError(f"Phase 6.5A H0 identity mismatch: {key}")
                arrays[key] = np.asarray(data[key])
            for branch in BRANCH_ORDER:
                value = np.asarray(h0[f"{split}_{branch}"], dtype=np.float32)
                if value.shape != (len(sequences[split]), BRANCH_DIMS[branch]):
                    raise ValueError(f"invalid Phase 6.5A H0 branch: {split}_{branch}")
                arrays[f"{split}_{branch}"] = value
    checkpoint_provenance: dict[str, Any] = {}
    for variant in VARIANTS:
        checkpoint_path = _checkpoint_path(encoder_root, walk, variant)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        validation = validate_lstm_capacity_encoder(
            encoder_dataset_path, checkpoint_path.parents[1]
        )
        extraction_seconds: dict[str, float] = {}
        for split in ("train", "test"):
            started = time.perf_counter()
            arrays[f"{split}_{variant}"] = extract_lstm2_branch(
                sequences[split],
                checkpoint_path,
                walk=walk,
                variant=variant,
                device=device,
                batch_size=batch_size,
            )
            extraction_seconds[split] = time.perf_counter() - started
        checkpoint_provenance[variant] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "validation": validation,
            "extraction_seconds": extraction_seconds,
        }
    branch_hashes = {
        split: {
            branch: array_sha256(arrays[f"{split}_{branch}"])
            for branch in MASTER_BRANCH_DIMS
        }
        for split in ("train", "test")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "6.5A",
        "task": task,
        "walk": walk,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
        "h0_feature_path": str(h0_feature_path.resolve()),
        "h0_feature_sha256": sha256_file(h0_feature_path),
        "h0_feature_validation": h0_validation,
        "encoder_dataset_path": str(encoder_dataset_path.resolve()),
        "encoder_dataset_sha256": sha256_file(encoder_dataset_path),
        "row_counts": {split: int(len(sequences[split])) for split in ("train", "test")},
        "master_branch_dims": MASTER_BRANCH_DIMS,
        "configuration_branches": {key: list(value) for key, value in CONFIG_BRANCHES.items()},
        "configuration_dims": CONFIG_DIMS,
        "branch_hashes": branch_hashes,
        "checkpoint_provenance": checkpoint_provenance,
        "target_independent_extraction": True,
        "epoch": 50,
        "environment": environment_manifest(resolve_device(device)),
    }
    _atomic_savez(output_path, arrays)
    manifest["feature_store_sha256"] = sha256_file(output_path)
    write_json(manifest_path, manifest)
    return validate_lstm_capacity_feature_store(output_path)


def validate_lstm_capacity_feature_store(feature_path: Path) -> dict[str, Any]:
    feature_path = Path(feature_path)
    manifest_path = Path(f"{feature_path}.manifest.json")
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Phase 6.5A feature store and manifest are required")
    feature_stat = feature_path.stat()
    manifest_stat = manifest_path.stat()
    return _validate_feature_store_cached(
        str(feature_path.resolve()),
        feature_stat.st_size,
        feature_stat.st_mtime_ns,
        manifest_stat.st_size,
        manifest_stat.st_mtime_ns,
    )


@lru_cache(maxsize=12)
def _validate_feature_store_cached(
    feature_path_string: str,
    feature_size: int,
    feature_mtime_ns: int,
    manifest_size: int,
    manifest_mtime_ns: int,
) -> dict[str, Any]:
    del feature_size, feature_mtime_ns, manifest_size, manifest_mtime_ns
    feature_path = Path(feature_path_string)
    manifest = json.loads(Path(f"{feature_path}.manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected Phase 6.5A feature schema")
    if manifest.get("configuration_branches") != {
        key: list(value) for key, value in CONFIG_BRANCHES.items()
    } or manifest.get("configuration_dims") != CONFIG_DIMS:
        raise ValueError("Phase 6.5A feature configuration matrix mismatch")
    if sha256_file(feature_path) != manifest.get("feature_store_sha256"):
        raise ValueError("Phase 6.5A feature-store hash mismatch")
    dataset_path = Path(manifest["dataset_path"])
    h0_path = Path(manifest["h0_feature_path"])
    if sha256_file(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Phase 6.5A feature dataset hash mismatch")
    if sha256_file(Path(f"{dataset_path}.manifest.json")) != manifest["dataset_manifest_sha256"]:
        raise ValueError("Phase 6.5A feature dataset-manifest hash mismatch")
    if sha256_file(h0_path) != manifest["h0_feature_sha256"]:
        raise ValueError("Phase 6.5A H0 feature hash mismatch")
    _validate_h0_source(str(manifest["task"]), dataset_path, h0_path)
    encoder_dataset_path = Path(manifest["encoder_dataset_path"])
    if sha256_file(encoder_dataset_path) != manifest["encoder_dataset_sha256"]:
        raise ValueError("Phase 6.5A encoder dataset hash mismatch")
    provenance = manifest.get("checkpoint_provenance", {})
    if set(provenance) != set(VARIANTS):
        raise ValueError("Phase 6.5A checkpoint inventory mismatch")
    for variant, source in provenance.items():
        checkpoint_path = Path(source["path"])
        if sha256_file(checkpoint_path) != source["sha256"]:
            raise ValueError(f"Phase 6.5A checkpoint hash mismatch: {variant}")
        validate_lstm_capacity_encoder(encoder_dataset_path, checkpoint_path.parents[1])
    with np.load(feature_path, allow_pickle=False) as values, np.load(
        dataset_path, allow_pickle=False
    ) as data, np.load(h0_path, allow_pickle=False) as h0:
        expected = {
            f"{split}_{field}"
            for split in ("train", "test")
            for field in IDENTITY_FIELDS
        } | {
            f"{split}_{branch}"
            for split in ("train", "test")
            for branch in MASTER_BRANCH_DIMS
        }
        if set(values.files) != expected:
            raise ValueError("Phase 6.5A feature store has missing or unexpected arrays")
        for split in ("train", "test"):
            count = int(manifest["row_counts"][split])
            for field in IDENTITY_FIELDS:
                key = f"{split}_{field}"
                if not np.array_equal(values[key], data[key]) or not np.array_equal(values[key], h0[key]):
                    raise ValueError(f"Phase 6.5A feature identity mismatch: {key}")
            for branch, width in MASTER_BRANCH_DIMS.items():
                key = f"{split}_{branch}"
                value = np.asarray(values[key])
                if value.shape != (count, width) or value.dtype != np.float32:
                    raise ValueError(f"invalid Phase 6.5A branch shape/dtype: {key}")
                if not np.isfinite(value).all() or array_sha256(value) != manifest["branch_hashes"][split][branch]:
                    raise ValueError(f"invalid Phase 6.5A branch values/hash: {key}")
            for branch in BRANCH_ORDER:
                if not np.array_equal(values[f"{split}_{branch}"], h0[f"{split}_{branch}"]):
                    raise ValueError(f"Phase 6.5A H0 branch changed: {split}_{branch}")
    return {
        "valid": True,
        "task": manifest["task"],
        "walk": int(manifest["walk"]),
        "row_counts": dict(manifest["row_counts"]),
        "configuration_dims": dict(CONFIG_DIMS),
        "target_independent": True,
    }


def load_lstm_capacity_configuration(
    feature_path: Path, configuration: str
) -> tuple[np.ndarray, np.ndarray]:
    validate_lstm_capacity_feature_store(feature_path)
    if configuration not in CONFIG_BRANCHES:
        raise ValueError(f"unknown Phase 6.5A configuration: {configuration}")
    with np.load(feature_path, allow_pickle=False) as stored:
        branches = CONFIG_BRANCHES[configuration]
        train = np.concatenate([stored[f"train_{branch}"] for branch in branches], axis=1)
        test = np.concatenate([stored[f"test_{branch}"] for branch in branches], axis=1)
    width = CONFIG_DIMS[configuration]
    if train.shape[1] != width or test.shape[1] != width:
        raise ValueError("Phase 6.5A configured feature width mismatch")
    return train.astype(np.float32, copy=False), test.astype(np.float32, copy=False)
