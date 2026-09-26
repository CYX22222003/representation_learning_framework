"""Diagnostics and aggregation helpers for Phase 6 encoder variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.phase5_walks import sha256_arrays, sha256_file
from features.phase5_features import extract_neural_features
from features.phase6_encoder_variant_features import extract_temporal_branch
from training.phase5_encoder import environment_manifest, resolve_device, write_json


CKA_SAMPLE_SIZE = 4096


def cka_sample_contract(dataset_path: Path) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=False) as stored:
        count = min(CKA_SAMPLE_SIZE, len(stored["encoder_train_sequences"]))
        fields = (
            np.asarray(stored["encoder_train_condition_ids"][:count]),
            np.asarray(stored["encoder_train_window_start_ns"][:count]),
            np.asarray(stored["encoder_train_decision_date_ns"][:count]),
            np.asarray(stored["encoder_train_decision_availability_ns"][:count]),
        )
    return {
        "selection": "first_rows_in_saved_encoder_train_order",
        "sample_size": count,
        "identity_sha256": sha256_arrays(*fields),
    }


def centered_linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or len(x) < 2:
        raise ValueError("CKA inputs must be aligned non-trivial matrices")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if denominator == 0.0:
        raise ValueError("CKA is undefined for a collapsed representation")
    return float(cross / denominator)


def analyze_walk_cka(
    dataset_path: Path,
    canonical_encoder_root: Path,
    variant_root: Path,
    output_path: Path,
    *,
    walk: int,
    device: str = "cuda",
    batch_size: int = 1024,
) -> dict[str, Any]:
    contract = cka_sample_contract(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as stored:
        sequences = np.asarray(
            stored["encoder_train_sequences"][: contract["sample_size"]], dtype=np.float32
        )
    resolved = resolve_device(device)
    canonical: dict[str, np.ndarray] = {}
    for family in ("contrastive", "byol"):
        checkpoint = canonical_encoder_root / f"walk{walk}" / family / "seed0" / "e50" / "checkpoint.pth"
        canonical[family] = extract_neural_features(
            sequences, checkpoint, family, device=resolved, batch_size=batch_size
        )
    rows = []
    for family in ("contrastive", "byol"):
        for backbone in ("lstm", "transformer"):
            variant = f"{family}_{backbone}"
            checkpoint = variant_root / "pretraining" / f"walk{walk}" / family / backbone / "seed0" / "e50" / "checkpoint.pth"
            canonical_checkpoint = (
                canonical_encoder_root
                / f"walk{walk}"
                / family
                / "seed0"
                / "e50"
                / "checkpoint.pth"
            )
            temporal = extract_temporal_branch(
                sequences, checkpoint, walk=walk, variant=variant,
                device=device, batch_size=batch_size,
            )
            rows.append(
                {
                    "walk": walk,
                    "family": family,
                    "variant": variant,
                    "cka": centered_linear_cka(canonical[family], temporal),
                    "canonical_checkpoint_path": str(canonical_checkpoint.resolve()),
                    "canonical_checkpoint_sha256": sha256_file(canonical_checkpoint),
                    "temporal_checkpoint_path": str(checkpoint.resolve()),
                    "temporal_checkpoint_sha256": sha256_file(checkpoint),
                }
            )
    payload = {
        "phase": 6,
        "purpose": "descriptive_temporal_vs_family_cnn_linear_cka",
        "walk": walk,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "sample_contract": contract,
        "selection_or_checkpoint_choice_performed": False,
        "environment": environment_manifest(resolved),
        "rows": rows,
    }
    write_json(output_path, payload)
    return payload


def validate_cka_report(path: Path, dataset_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("CKA dataset hash mismatch")
    if payload.get("sample_contract") != cka_sample_contract(dataset_path):
        raise ValueError("CKA sample contract mismatch")
    rows = payload.get("rows", [])
    if len(rows) != 4 or any(not 0.0 <= float(row["cka"]) <= 1.0 + 1e-10 for row in rows):
        raise ValueError("CKA report does not contain four valid comparisons")
    expected = {
        (family, f"{family}_{backbone}")
        for family in ("contrastive", "byol")
        for backbone in ("lstm", "transformer")
    }
    if {(row["family"], row["variant"]) for row in rows} != expected:
        raise ValueError("CKA comparison inventory mismatch")
    for row in rows:
        for prefix in ("canonical", "temporal"):
            checkpoint = Path(row[f"{prefix}_checkpoint_path"])
            if sha256_file(checkpoint) != row[f"{prefix}_checkpoint_sha256"]:
                raise ValueError(f"CKA {prefix} checkpoint hash mismatch")
    return {"valid": True, "walk": int(payload["walk"]), "comparisons": len(rows)}
