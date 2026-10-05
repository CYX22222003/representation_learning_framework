"""Minimal data dependency contract for the Phase 6.7 TimeDART extension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_processing.phase5_walks import sha256_arrays, sha256_file
from features.phase5_features import IDENTITY_FIELDS, array_sha256
from training.phase5_encoder import write_json


METHOD = "timedart_frozen"
TASKS = ("classification_h2", "absolute_price_h8", "realised_variance")
SCHEMA_VERSION = "phase6-7-timedart-data-v1"


def timedart_dataset_paths(project_root: Path) -> dict[int, dict[str, Path]]:
    """Return only the six existing bundles needed by TimeDART."""

    root = Path(project_root)
    return {
        walk: {
            "classification_h2": root
            / "experiments"
            / "phase5"
            / "data_preparation"
            / f"walk{walk}"
            / "market_1h_seq64_h2.npz",
            "absolute_price_h8": root
            / "experiments"
            / "phase5"
            / "downstream_addons"
            / "shared"
            / "h8"
            / "data"
            / f"walk{walk}"
            / "market_1h_seq64_h8.npz",
            "realised_variance": root
            / "experiments"
            / "phase6"
            / "volatility_prediction"
            / "data_preparation"
            / f"walk{walk}"
            / "volatility_1h_seq64_h8.npz",
        }
        for walk in (1, 2)
    }


def encoder_dataset_path(project_root: Path, walk: int) -> Path:
    if walk not in (1, 2):
        raise ValueError("walk must be 1 or 2")
    return timedart_dataset_paths(project_root)[walk]["classification_h2"]


def _source_manifest_hash(path: Path) -> str | None:
    manifest = Path(f"{path}.manifest.json")
    return sha256_file(manifest) if manifest.is_file() else None


def load_timedart_encoder_population(
    dataset_path: Path, walk: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load target-free encoder inputs and identity columns only."""

    path = Path(dataset_path)
    with np.load(path, allow_pickle=False) as stored:
        required = {
            "encoder_train_sequences",
            *(f"encoder_train_{field}" for field in IDENTITY_FIELDS),
        }
        missing = sorted(required.difference(stored.files))
        if missing:
            raise ValueError(f"TimeDART encoder bundle is missing arrays: {missing}")
        sequences = np.asarray(stored["encoder_train_sequences"], dtype=np.float32)
        identities = {
            field: np.asarray(stored[f"encoder_train_{field}"])
            for field in IDENTITY_FIELDS
        }
    if sequences.ndim != 3 or sequences.shape[1:] != (64, 5):
        raise ValueError("TimeDART encoder inputs must have shape [N,64,5]")
    if len(sequences) == 0 or not np.isfinite(sequences).all():
        raise ValueError("TimeDART encoder inputs must be non-empty and finite")
    if any(len(value) != len(sequences) for value in identities.values()):
        raise ValueError("TimeDART encoder identities are not row-aligned")
    identity_hash = sha256_arrays(*(identities[field] for field in IDENTITY_FIELDS))
    source_manifest_path = Path(f"{path}.manifest.json")
    if source_manifest_path.is_file():
        source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        expected_walk = source.get("walk")
        if expected_walk is not None and int(expected_walk) != walk:
            raise ValueError("TimeDART encoder bundle walk mismatch")
        expected_hash = source.get("identity_hashes", {}).get("encoder_train")
        if expected_hash is not None and expected_hash != identity_hash:
            raise ValueError("TimeDART encoder identity hash mismatch")
    return sequences, {
        "walk": walk,
        "rows": len(sequences),
        "sequence_hash": array_sha256(sequences),
        "identity_hash": identity_hash,
        "dataset_sha256": sha256_file(path),
        "dataset_manifest_sha256": _source_manifest_hash(path),
        "arrays_loaded": [
            "encoder_train_sequences",
            *(f"encoder_train_{field}" for field in IDENTITY_FIELDS),
        ],
        "targets_loaded": False,
        "evaluation_rows_loaded": False,
    }


def load_timedart_task_contexts(
    dataset_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load task contexts and identities, deliberately excluding every target."""

    path = Path(dataset_path)
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as stored:
        required = {
            *(f"{split}_sequences" for split in ("train", "test")),
            *(
                f"{split}_{field}"
                for split in ("train", "test")
                for field in IDENTITY_FIELDS
            ),
        }
        missing = sorted(required.difference(stored.files))
        if missing:
            raise ValueError(f"TimeDART task bundle is missing arrays: {missing}")
        for split in ("train", "test"):
            arrays[f"{split}_sequences"] = np.asarray(
                stored[f"{split}_sequences"], dtype=np.float32
            )
            for field in IDENTITY_FIELDS:
                arrays[f"{split}_{field}"] = np.asarray(stored[f"{split}_{field}"])
    split_records: dict[str, Any] = {}
    for split in ("train", "test"):
        sequences = arrays[f"{split}_sequences"]
        if sequences.ndim != 3 or sequences.shape[1:] != (64, 5):
            raise ValueError(f"TimeDART {split} contexts must have shape [N,64,5]")
        if len(sequences) == 0 or not np.isfinite(sequences).all():
            raise ValueError(f"TimeDART {split} contexts must be non-empty and finite")
        identities = [arrays[f"{split}_{field}"] for field in IDENTITY_FIELDS]
        if any(len(value) != len(sequences) for value in identities):
            raise ValueError(f"TimeDART {split} identities are not row-aligned")
        split_records[split] = {
            "rows": len(sequences),
            "sequence_hash": array_sha256(sequences),
            "identity_hash": sha256_arrays(*identities),
        }
    return arrays, {
        "dataset_sha256": sha256_file(path),
        "dataset_manifest_sha256": _source_manifest_hash(path),
        "splits": split_records,
        "targets_loaded": False,
        "arrays_loaded": sorted(arrays),
    }


def prepare_timedart_data_manifest(project_root: Path, output_path: Path) -> dict[str, Any]:
    """Freeze hashes and relevant row contracts without copying source bundles."""

    paths = timedart_dataset_paths(project_root)
    walks: dict[str, Any] = {}
    for walk in (1, 2):
        encoder_values, encoder = load_timedart_encoder_population(
            paths[walk]["classification_h2"], walk
        )
        del encoder_values
        tasks: dict[str, Any] = {}
        for task in TASKS:
            contexts, record = load_timedart_task_contexts(paths[walk][task])
            del contexts
            tasks[task] = {
                "path": str(paths[walk][task].resolve()),
                **record,
            }
        walks[f"walk{walk}"] = {
            "encoder_dataset_path": str(paths[walk]["classification_h2"].resolve()),
            "encoder": encoder,
            "tasks": tasks,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": "6.7",
        "method": METHOD,
        "unique_source_file_count": 6,
        "source_bundles_copied": False,
        "only_relevant_arrays_loaded": True,
        "walks": walks,
    }
    output = Path(output_path)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing TimeDART data manifest differs from current inputs")
    else:
        write_json(output, payload)
    return payload

