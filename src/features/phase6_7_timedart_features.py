"""TimeDART frozen master feature stores for the Phase 6.7 probes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data_processing.phase5_walks import sha256_arrays, sha256_file
from data_processing.phase6_7_timedart_data import (
    METHOD,
    TASKS,
    load_timedart_task_contexts,
)
from features.phase5_features import IDENTITY_FIELDS, array_sha256
from training.phase5_encoder import environment_manifest, resolve_device, write_json
from training.phase6_7_timedart import (
    _replay_diagnostics,
    load_timedart_encoder_checkpoint,
    validate_timedart_pretraining,
)


OUTPUT_DIM = 170
SCHEMA_VERSION = "phase6-7-timedart-master-features-v1"
COORDINATE_SLICES = {
    "open_pooled": [0, 32],
    "high_pooled": [32, 64],
    "low_pooled": [64, 96],
    "close_pooled": [96, 128],
    "volume_pooled": [128, 160],
    "window_means_ohlcv": [160, 165],
    "window_stds_ohlcv": [165, 170],
}


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


@torch.no_grad()
def extract_timedart_embeddings(
    sequences: np.ndarray,
    checkpoint_path: Path,
    *,
    walk: int,
    device: str = "cuda",
    batch_size: int = 1024,
) -> np.ndarray:
    """Run only the retained unmasked clean encoder and return 170 coordinates."""

    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (64, 5):
        raise ValueError("TimeDART extraction requires [N,64,5] contexts")
    if len(values) == 0 or not np.isfinite(values).all() or batch_size <= 0:
        raise ValueError("TimeDART extraction inputs must be finite and batch size positive")
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    encoder, _ = load_timedart_encoder_checkpoint(
        checkpoint_path, walk=walk, device=resolved
    )
    outputs = []
    tensor = torch.from_numpy(values)
    for batch in tensor.split(batch_size):
        outputs.append(encoder.extract(batch.to(resolved)).cpu().numpy())
    features = np.concatenate(outputs).astype(np.float32, copy=False)
    if features.shape != (len(values), OUTPUT_DIM) or not np.isfinite(features).all():
        raise ValueError("TimeDART feature extraction returned invalid output")
    return features


def build_timedart_feature_store(
    task_datasets: Mapping[str, Path],
    encoder_dataset_path: Path,
    encoder_run_root: Path,
    output_path: Path,
    *,
    walk: int,
    device: str = "cuda",
    batch_size: int = 1024,
) -> dict[str, Any]:
    if walk not in (1, 2) or set(task_datasets) != set(TASKS):
        raise ValueError("TimeDART store requires the three declared tasks for one walk")
    manifest_path = Path(f"{output_path}.manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite TimeDART feature store: {output_path}")
    source_validation: dict[str, Any]
    try:
        source_validation = validate_timedart_pretraining(
            encoder_dataset_path, encoder_run_root
        )
    except Exception as exc:
        source_validation = {
            "valid": False,
            "status": "warning",
            "message": str(exc),
            "continued": True,
        }
    checkpoint_path = encoder_run_root / "e50" / "encoder.pth"
    # Loading this checkpoint is a functional dependency, not an advisory
    # validator. A missing/corrupt checkpoint cannot produce representations.
    _, checkpoint = load_timedart_encoder_checkpoint(
        checkpoint_path, walk=walk, device="cpu"
    )

    saved: dict[str, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    for task in TASKS:
        dataset_path = Path(task_datasets[task])
        contexts, source = load_timedart_task_contexts(dataset_path)
        split_records: dict[str, Any] = {}
        for split in ("train", "test"):
            values = contexts[f"{split}_sequences"]
            started = time.perf_counter()
            features = extract_timedart_embeddings(
                values,
                checkpoint_path,
                walk=walk,
                device=device,
                batch_size=batch_size,
            )
            prefix = f"{task}_{split}"
            saved[f"{prefix}_features"] = features
            for field in IDENTITY_FIELDS:
                saved[f"{prefix}_{field}"] = contexts[f"{split}_{field}"]
            split_records[split] = {
                "rows": len(features),
                "feature_hash": array_sha256(features),
                "sequence_hash": source["splits"][split]["sequence_hash"],
                "identity_hash": source["splits"][split]["identity_hash"],
                "extraction_seconds": time.perf_counter() - started,
            }
        provenance[task] = {
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": source["dataset_sha256"],
            "dataset_manifest_sha256": source["dataset_manifest_sha256"],
            "targets_loaded_during_extraction": False,
            "arrays_loaded": source["arrays_loaded"],
            "splits": split_records,
        }

    probe_task = TASKS[0]
    probe_values, _ = load_timedart_task_contexts(Path(task_datasets[probe_task]))
    probe = probe_values["train_sequences"][:8]
    cpu = extract_timedart_embeddings(
        probe, checkpoint_path, walk=walk, device="cpu", batch_size=8
    )
    other = extract_timedart_embeddings(
        probe, checkpoint_path, walk=walk, device=device, batch_size=8
    )
    cross_device = _replay_diagnostics(torch.from_numpy(other), torch.from_numpy(cpu))
    if not cross_device["accepted"]:
        raise ValueError("TimeDART feature CPU/device probe has material drift")
    batching = extract_timedart_embeddings(
        probe, checkpoint_path, walk=walk, device=device, batch_size=1
    )
    if not np.allclose(other, batching, rtol=1e-6, atol=1e-6):
        raise ValueError("TimeDART extraction is not batching-invariant")

    _atomic_savez(output_path, saved)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "6.7",
        "method": METHOD,
        "walk": walk,
        "output_dim": OUTPUT_DIM,
        "task_order": list(TASKS),
        "coordinate_slices": COORDINATE_SLICES,
        "encoder_dataset_path": str(encoder_dataset_path.resolve()),
        "encoder_dataset_sha256": sha256_file(encoder_dataset_path),
        "encoder_run_root": str(encoder_run_root.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_encoder_state_only_at_extraction": True,
        "decoder_diffusion_projector_constructed_or_executed": False,
        "source_checkpoint_validation": source_validation,
        "task_provenance": provenance,
        "cpu_device_probe": cross_device,
        "batching_invariant_probe": True,
        "targets_loaded_during_extraction": False,
        "checkpoint_completed_epoch": int(checkpoint["completed_epoch"]),
        "environment": environment_manifest(resolve_device(device)),
    }
    manifest["feature_store_sha256"] = sha256_file(output_path)
    write_json(manifest_path, manifest)
    try:
        return validate_timedart_feature_store(output_path, replay=False)
    except Exception as exc:
        warning = {
            "valid": False,
            "status": "warning",
            "continued": True,
            "method": METHOD,
            "walk": walk,
            "message": str(exc),
        }
        write_json(output_path.parent / "validation_warning.json", warning)
        return warning


def validate_timedart_feature_store(
    feature_path: Path,
    *,
    replay: bool = False,
    device: str = "cpu",
    batch_size: int = 1024,
) -> dict[str, Any]:
    path = Path(feature_path)
    manifest_path = Path(f"{path}.manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("TimeDART feature store or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("method") != METHOD
        or int(manifest.get("output_dim", -1)) != OUTPUT_DIM
        or manifest.get("coordinate_slices") != COORDINATE_SLICES
    ):
        raise ValueError("TimeDART feature manifest contract mismatch")
    if manifest.get("feature_store_sha256") != sha256_file(path):
        raise ValueError("TimeDART feature store hash mismatch")
    walk = int(manifest["walk"])
    with np.load(path, allow_pickle=False) as stored:
        expected_fields = {
            *(f"{task}_{split}_features" for task in TASKS for split in ("train", "test")),
            *(
                f"{task}_{split}_{field}"
                for task in TASKS
                for split in ("train", "test")
                for field in IDENTITY_FIELDS
            ),
        }
        if set(stored.files) != expected_fields:
            raise ValueError("TimeDART feature store field inventory mismatch")
        for task in TASKS:
            record = manifest["task_provenance"][task]
            dataset = Path(record["dataset_path"])
            contexts, source = load_timedart_task_contexts(dataset)
            if record["dataset_sha256"] != sha256_file(dataset):
                raise ValueError(f"TimeDART {task} dataset hash mismatch")
            for split in ("train", "test"):
                prefix = f"{task}_{split}"
                features = np.asarray(stored[f"{prefix}_features"])
                if features.shape != (len(contexts[f"{split}_sequences"]), OUTPUT_DIM):
                    raise ValueError(f"TimeDART {prefix} feature shape mismatch")
                if not np.isfinite(features).all():
                    raise ValueError(f"TimeDART {prefix} has non-finite features")
                if record["splits"][split]["feature_hash"] != array_sha256(features):
                    raise ValueError(f"TimeDART {prefix} feature hash mismatch")
                for field in IDENTITY_FIELDS:
                    if not np.array_equal(
                        stored[f"{prefix}_{field}"], contexts[f"{split}_{field}"]
                    ):
                        raise ValueError(f"TimeDART {prefix} identity mismatch: {field}")
                if record["splits"][split]["identity_hash"] != sha256_arrays(
                    *(contexts[f"{split}_{field}"] for field in IDENTITY_FIELDS)
                ):
                    raise ValueError(f"TimeDART {prefix} identity hash mismatch")
                if replay:
                    actual = extract_timedart_embeddings(
                        contexts[f"{split}_sequences"],
                        Path(manifest["checkpoint_path"]),
                        walk=walk,
                        device=device,
                        batch_size=batch_size,
                    )
                    diagnostics = _replay_diagnostics(
                        torch.from_numpy(actual), torch.from_numpy(features)
                    )
                    if not diagnostics["accepted"]:
                        raise ValueError(f"TimeDART {prefix} material feature replay drift")
    return {
        "valid": True,
        "status": (
            "warning"
            if not manifest["cpu_device_probe"].get("strict", False)
            else "pass"
        ),
        "method": METHOD,
        "walk": walk,
        "output_dim": OUTPUT_DIM,
        "tasks": list(TASKS),
        "replayed": replay,
    }


def load_timedart_task_features(
    feature_path: Path, task: str, *, validate: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if task not in TASKS:
        raise ValueError(f"unknown TimeDART task: {task}")
    if validate:
        validate_timedart_feature_store(feature_path, replay=False)
    with np.load(feature_path, allow_pickle=False) as stored:
        train = np.asarray(stored[f"{task}_train_features"], dtype=np.float32)
        test = np.asarray(stored[f"{task}_test_features"], dtype=np.float32)
    return train, test
