"""Descriptive CKA diagnostics for Phase 6.5D residual-CNN encoders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.phase5_walks import sha256_file
from evaluation.phase6_encoder_variants import centered_linear_cka, cka_sample_contract
from features.phase5_features import extract_neural_features
from features.phase6_5_residual_cnn_features import extract_residual_cnn_branch
from training.phase5_encoder import environment_manifest, resolve_device, write_json


def analyze_residual_cnn_cka(
    dataset_path: Path,
    canonical_encoder_root: Path,
    residual_root: Path,
    output_path: Path,
    *,
    walk: int,
    device: str = "cuda",
    batch_size: int = 1024,
) -> dict[str, Any]:
    contract = cka_sample_contract(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as stored:
        sequences = np.asarray(
            stored["encoder_train_sequences"][: contract["sample_size"]],
            dtype=np.float32,
        )
    resolved = resolve_device(device)
    rows = []
    for family in ("contrastive", "byol"):
        canonical_checkpoint = (
            canonical_encoder_root
            / f"walk{walk}"
            / family
            / "seed0"
            / "e50"
            / "checkpoint.pth"
        )
        candidate_checkpoint = (
            residual_root
            / "pretraining"
            / f"walk{walk}"
            / family
            / "rescnn"
            / "seed0"
            / "e50"
            / "checkpoint.pth"
        )
        canonical = extract_neural_features(
            sequences,
            canonical_checkpoint,
            family,
            device=resolved,
            batch_size=batch_size,
        )
        candidate = extract_residual_cnn_branch(
            sequences,
            candidate_checkpoint,
            walk=walk,
            variant=f"{family}_rescnn",
            device=device,
            batch_size=batch_size,
        )
        rows.append(
            {
                "walk": walk,
                "family": family,
                "candidate": f"{family}_rescnn",
                "reference": f"{family}_canonical_cnn",
                "cka": centered_linear_cka(candidate, canonical),
                "candidate_checkpoint_path": str(candidate_checkpoint.resolve()),
                "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
                "reference_checkpoint_path": str(canonical_checkpoint.resolve()),
                "reference_checkpoint_sha256": sha256_file(canonical_checkpoint),
            }
        )
    payload = {
        "phase": "6.5D",
        "purpose": "descriptive_residual_cnn_vs_same_family_canonical_cnn_linear_cka",
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


def validate_residual_cnn_cka(path: Path, dataset_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != "6.5D":
        raise ValueError("Phase 6.5D CKA phase mismatch")
    if payload.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("Phase 6.5D CKA dataset hash mismatch")
    if payload.get("sample_contract") != cka_sample_contract(dataset_path):
        raise ValueError("Phase 6.5D CKA sample contract mismatch")
    rows = payload.get("rows", [])
    expected = {
        (family, f"{family}_rescnn", f"{family}_canonical_cnn")
        for family in ("contrastive", "byol")
    }
    if len(rows) != 2 or {
        (row["family"], row["candidate"], row["reference"]) for row in rows
    } != expected:
        raise ValueError("Phase 6.5D CKA inventory mismatch")
    if any(not 0.0 <= float(row["cka"]) <= 1.0 + 1e-10 for row in rows):
        raise ValueError("Phase 6.5D CKA contains an invalid value")
    for row in rows:
        for prefix in ("candidate", "reference"):
            checkpoint = Path(row[f"{prefix}_checkpoint_path"])
            if sha256_file(checkpoint) != row[f"{prefix}_checkpoint_sha256"]:
                raise ValueError(f"Phase 6.5D CKA {prefix} checkpoint hash mismatch")
    return {"valid": True, "walk": int(payload["walk"]), "comparisons": len(rows)}
