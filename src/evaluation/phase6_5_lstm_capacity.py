"""Descriptive CKA diagnostics for Phase 6.5A two-layer LSTM encoders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.phase5_walks import sha256_file
from evaluation.phase6_encoder_variants import (
    centered_linear_cka,
    cka_sample_contract,
)
from features.phase5_features import extract_neural_features
from features.phase6_5_lstm_capacity_features import extract_lstm2_branch
from features.phase6_encoder_variant_features import extract_temporal_branch
from training.phase5_encoder import environment_manifest, resolve_device, write_json


def analyze_lstm_capacity_cka(
    dataset_path: Path,
    canonical_encoder_root: Path,
    phase6_variant_root: Path,
    phase65_capacity_root: Path,
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
        shallow_checkpoint = (
            phase6_variant_root
            / "pretraining"
            / f"walk{walk}"
            / family
            / "lstm"
            / "seed0"
            / "e50"
            / "checkpoint.pth"
        )
        deep_checkpoint = (
            phase65_capacity_root
            / "pretraining"
            / f"walk{walk}"
            / family
            / "lstm2"
            / "seed0"
            / "e50"
            / "checkpoint.pth"
        )
        cnn = extract_neural_features(
            sequences,
            canonical_checkpoint,
            family,
            device=resolved,
            batch_size=batch_size,
        )
        shallow = extract_temporal_branch(
            sequences,
            shallow_checkpoint,
            walk=walk,
            variant=f"{family}_lstm",
            device=device,
            batch_size=batch_size,
        )
        deep = extract_lstm2_branch(
            sequences,
            deep_checkpoint,
            walk=walk,
            variant=f"{family}_lstm2",
            device=device,
            batch_size=batch_size,
        )
        for reference_name, reference, reference_checkpoint in (
            ("lstm1", shallow, shallow_checkpoint),
            ("cnn", cnn, canonical_checkpoint),
        ):
            rows.append(
                {
                    "walk": walk,
                    "family": family,
                    "candidate": f"{family}_lstm2",
                    "reference": reference_name,
                    "cka": centered_linear_cka(deep, reference),
                    "candidate_checkpoint_path": str(deep_checkpoint.resolve()),
                    "candidate_checkpoint_sha256": sha256_file(deep_checkpoint),
                    "reference_checkpoint_path": str(reference_checkpoint.resolve()),
                    "reference_checkpoint_sha256": sha256_file(reference_checkpoint),
                }
            )
    payload = {
        "phase": "6.5A",
        "purpose": "descriptive_lstm2_vs_lstm1_and_family_cnn_linear_cka",
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


def validate_lstm_capacity_cka(path: Path, dataset_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != "6.5A":
        raise ValueError("Phase 6.5A CKA phase mismatch")
    if payload.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("Phase 6.5A CKA dataset hash mismatch")
    if payload.get("sample_contract") != cka_sample_contract(dataset_path):
        raise ValueError("Phase 6.5A CKA sample contract mismatch")
    rows = payload.get("rows", [])
    expected = {
        (family, f"{family}_lstm2", reference)
        for family in ("contrastive", "byol")
        for reference in ("lstm1", "cnn")
    }
    if len(rows) != 4 or {
        (row["family"], row["candidate"], row["reference"]) for row in rows
    } != expected:
        raise ValueError("Phase 6.5A CKA inventory mismatch")
    if any(not 0.0 <= float(row["cka"]) <= 1.0 + 1e-10 for row in rows):
        raise ValueError("Phase 6.5A CKA contains an invalid value")
    for row in rows:
        for prefix in ("candidate", "reference"):
            checkpoint = Path(row[f"{prefix}_checkpoint_path"])
            if sha256_file(checkpoint) != row[f"{prefix}_checkpoint_sha256"]:
                raise ValueError(f"Phase 6.5A CKA {prefix} checkpoint hash mismatch")
    return {"valid": True, "walk": int(payload["walk"]), "comparisons": len(rows)}
