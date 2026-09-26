"""Identity-safe alignment of frozen Phase 5 features to Phase 6 volatility rows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_processing.phase5_walks import sha256_file
from data_processing.phase6_volatility_labels import validate_volatility_label_bundle_files
from features.phase5_features import (
    BRANCH_DIMS,
    BRANCH_ORDER,
    IDENTITY_FIELDS,
    array_sha256,
    validate_phase5_feature_bundle,
)
from training.phase5_encoder import write_json


PHASE6_VOLATILITY_FEATURE_SCHEMA_VERSION = "phase6-volatility-feature-v1"
SPLITS = ("train", "test")


def identity_mapping(
    source: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray], split: str
) -> np.ndarray:
    """Map unique target identities into a unique superset source population."""

    source_arrays = [np.asarray(source[f"{split}_{field}"]) for field in IDENTITY_FIELDS]
    target_arrays = [np.asarray(target[f"{split}_{field}"]) for field in IDENTITY_FIELDS]
    source_identities = list(zip(*(value.tolist() for value in source_arrays), strict=True))
    target_identities = list(zip(*(value.tolist() for value in target_arrays), strict=True))
    if len(set(source_identities)) != len(source_identities):
        raise ValueError(f"{split} source feature identities are not unique")
    if len(set(target_identities)) != len(target_identities):
        raise ValueError(f"{split} Phase 6 label identities are not unique")
    lookup = {identity: index for index, identity in enumerate(source_identities)}
    try:
        return np.asarray([lookup[identity] for identity in target_identities], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"{split} Phase 6 identity is absent from frozen feature source") from error


def build_phase6_volatility_feature_bundle(
    label_path: Path,
    phase5_feature_path: Path,
    phase5_dataset_path: Path,
    output_path: Path,
    *,
    walk: int,
) -> dict[str, Any]:
    """Select byte-identical frozen feature rows for the Phase 6 population."""

    if output_path.exists() or Path(f"{output_path}.manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite Phase 6 feature bundle: {output_path}")
    label_validation = validate_volatility_label_bundle_files(label_path)
    feature_validation = validate_phase5_feature_bundle(
        phase5_feature_path, dataset_path=phase5_dataset_path
    )
    if int(label_validation["walk"]) != walk or int(feature_validation["walk"]) != walk:
        raise ValueError("walk mismatch across Phase 6 labels and Phase 5 features")

    with np.load(label_path, allow_pickle=False) as labels_stored, np.load(
        phase5_dataset_path, allow_pickle=False
    ) as phase5_data_stored, np.load(phase5_feature_path, allow_pickle=False) as features_stored:
        labels = {name: np.asarray(labels_stored[name]) for name in labels_stored.files}
        phase5_data = {
            name: np.asarray(phase5_data_stored[name]) for name in phase5_data_stored.files
        }
        source_features = {
            name: np.asarray(features_stored[name]) for name in features_stored.files
        }

    arrays: dict[str, np.ndarray] = {}
    source_indices: dict[str, np.ndarray] = {}
    for split in SPLITS:
        positions = identity_mapping(phase5_data, labels, split)
        source_indices[split] = positions
        if not np.array_equal(
            phase5_data[f"{split}_sequences"][positions], labels[f"{split}_sequences"]
        ):
            raise ValueError(f"{split} mapped model-ready contexts are not byte-identical")
        if not np.array_equal(
            phase5_data[f"{split}_raw_sequences"][positions], labels[f"{split}_raw_sequences"]
        ):
            raise ValueError(f"{split} mapped raw contexts are not byte-identical")
        for field in IDENTITY_FIELDS:
            arrays[f"{split}_{field}"] = np.asarray(labels[f"{split}_{field}"])
        arrays[f"{split}_shared_row_indices"] = np.asarray(
            labels[f"{split}_shared_row_indices"], dtype=np.int64
        )
        arrays[f"{split}_phase5_source_indices"] = positions
        for branch in BRANCH_ORDER:
            arrays[f"{split}_{branch}"] = np.asarray(
                source_features[f"{split}_{branch}"][positions], dtype=np.float32
            )

    phase5_feature_manifest_path = Path(f"{phase5_feature_path}.manifest.json")
    phase5_feature_manifest = json.loads(
        phase5_feature_manifest_path.read_text(encoding="utf-8")
    )
    label_manifest_path = Path(f"{label_path}.manifest.json")
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": PHASE6_VOLATILITY_FEATURE_SCHEMA_VERSION,
        "phase": 6,
        "stage": "frozen_canonical_feature_alignment",
        "purpose": "canonical_five_branch_features_for_future_realised_variance",
        "walk": walk,
        "row_counts": {
            split: int(len(arrays[f"{split}_shared_row_indices"])) for split in SPLITS
        },
        "branch_order": list(BRANCH_ORDER),
        "branch_dims": BRANCH_DIMS,
        "concat_dim": int(sum(BRANCH_DIMS.values())),
        "identity_array_hashes": {
            split: {
                field: array_sha256(arrays[f"{split}_{field}"])
                for field in IDENTITY_FIELDS
            }
            for split in SPLITS
        },
        "shared_row_index_hashes": {
            split: array_sha256(arrays[f"{split}_shared_row_indices"])
            for split in SPLITS
        },
        "source_index_hashes": {
            split: array_sha256(arrays[f"{split}_phase5_source_indices"])
            for split in SPLITS
        },
        "branch_hashes": {
            split: {
                branch: array_sha256(arrays[f"{split}_{branch}"])
                for branch in BRANCH_ORDER
            }
            for split in SPLITS
        },
        "label_bundle": {
            "path": str(label_path.resolve()),
            "sha256": sha256_file(label_path),
            "manifest_path": str(label_manifest_path.resolve()),
            "manifest_sha256": sha256_file(label_manifest_path),
            "identity_hashes": label_manifest["identity_hashes"],
        },
        "phase5_feature_source": {
            "path": str(phase5_feature_path.resolve()),
            "sha256": sha256_file(phase5_feature_path),
            "manifest_path": str(phase5_feature_manifest_path.resolve()),
            "manifest_sha256": sha256_file(phase5_feature_manifest_path),
        },
        "phase5_dataset_source": {
            "path": str(phase5_dataset_path.resolve()),
            "sha256": sha256_file(phase5_dataset_path),
            "manifest_path": str(Path(f"{phase5_dataset_path}.manifest.json").resolve()),
            "manifest_sha256": sha256_file(Path(f"{phase5_dataset_path}.manifest.json")),
        },
        "checkpoint_provenance": phase5_feature_manifest["checkpoint_provenance"],
        "target_independence": {
            "encoder_weights_pretrained_on_target_free_phase5_encoder_population": True,
            "features_selected_by_context_identity_only": True,
            "target_values_read_by_feature_extractor": False,
            "evaluation_rows_used_to_fit_parameters": False,
            "all_aligned_contexts_byte_identical": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output_path)
    manifest["feature_bundle_sha256"] = sha256_file(output_path)
    write_json(Path(f"{output_path}.manifest.json"), manifest)
    return validate_phase6_volatility_feature_bundle(output_path)


def validate_phase6_volatility_feature_bundle(feature_path: Path) -> dict[str, Any]:
    """Replay label identities and byte-identical selection from frozen features."""

    feature_path = Path(feature_path)
    manifest_path = Path(f"{feature_path}.manifest.json")
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Phase 6 feature bundle and manifest are both required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PHASE6_VOLATILITY_FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 6 volatility feature schema")
    if manifest.get("branch_order") != list(BRANCH_ORDER) or manifest.get(
        "branch_dims"
    ) != BRANCH_DIMS:
        raise ValueError("Phase 6 feature branch contract mismatch")
    if int(manifest.get("concat_dim", -1)) != 445:
        raise ValueError("Phase 6 canonical concat feature dimension must be 445")
    if sha256_file(feature_path) != manifest.get("feature_bundle_sha256"):
        raise ValueError("Phase 6 feature bundle hash mismatch")

    label_record = manifest["label_bundle"]
    phase5_feature_record = manifest["phase5_feature_source"]
    phase5_dataset_record = manifest["phase5_dataset_source"]
    for record_name, record in (
        ("label bundle", label_record),
        ("Phase 5 feature source", phase5_feature_record),
        ("Phase 5 dataset source", phase5_dataset_record),
    ):
        path = Path(record["path"])
        source_manifest = Path(record["manifest_path"])
        if not path.is_file() or not source_manifest.is_file():
            raise FileNotFoundError(f"missing {record_name}")
        if sha256_file(path) != record["sha256"] or sha256_file(source_manifest) != record[
            "manifest_sha256"
        ]:
            raise ValueError(f"{record_name} hash mismatch")
    label_validation = validate_volatility_label_bundle_files(Path(label_record["path"]))
    phase5_validation = validate_phase5_feature_bundle(
        Path(phase5_feature_record["path"]),
        dataset_path=Path(phase5_dataset_record["path"]),
    )
    if int(label_validation["walk"]) != int(manifest["walk"]) or int(
        phase5_validation["walk"]
    ) != int(manifest["walk"]):
        raise ValueError("feature lineage walk mismatch")
    independence = manifest.get("target_independence", {})
    if not all(
        independence.get(key) is expected
        for key, expected in {
            "encoder_weights_pretrained_on_target_free_phase5_encoder_population": True,
            "features_selected_by_context_identity_only": True,
            "target_values_read_by_feature_extractor": False,
            "evaluation_rows_used_to_fit_parameters": False,
            "all_aligned_contexts_byte_identical": True,
        }.items()
    ):
        raise ValueError("target-independence assertions are incomplete")

    with np.load(feature_path, allow_pickle=False) as aligned_stored, np.load(
        label_record["path"], allow_pickle=False
    ) as labels_stored, np.load(phase5_feature_record["path"], allow_pickle=False) as source_stored, np.load(
        phase5_dataset_record["path"], allow_pickle=False
    ) as dataset_stored:
        aligned = {name: np.asarray(aligned_stored[name]) for name in aligned_stored.files}
        labels = {name: np.asarray(labels_stored[name]) for name in labels_stored.files}
        source = {name: np.asarray(source_stored[name]) for name in source_stored.files}
        dataset = {name: np.asarray(dataset_stored[name]) for name in dataset_stored.files}
    for split in SPLITS:
        count = int(manifest["row_counts"][split])
        positions = np.asarray(aligned[f"{split}_phase5_source_indices"], dtype=np.int64)
        if positions.shape != (count,) or np.any(positions < 0):
            raise ValueError(f"{split} Phase 5 source map is invalid")
        if array_sha256(positions) != manifest["source_index_hashes"][split]:
            raise ValueError(f"{split} Phase 5 source-map hash mismatch")
        shared = np.asarray(aligned[f"{split}_shared_row_indices"], dtype=np.int64)
        if not np.array_equal(shared, labels[f"{split}_shared_row_indices"]):
            raise ValueError(f"{split} shared comparator rows differ from labels")
        if array_sha256(shared) != manifest["shared_row_index_hashes"][split]:
            raise ValueError(f"{split} shared row hash mismatch")
        for context_name in ("sequences", "raw_sequences"):
            if not np.array_equal(
                labels[f"{split}_{context_name}"],
                dataset[f"{split}_{context_name}"][positions],
            ):
                raise ValueError(
                    f"{split} {context_name} is not byte-identical to the frozen source context"
                )
        for field in IDENTITY_FIELDS:
            value = np.asarray(aligned[f"{split}_{field}"])
            if not np.array_equal(value, labels[f"{split}_{field}"]):
                raise ValueError(f"{split} aligned {field} differs from label identities")
            if not np.array_equal(value, source[f"{split}_{field}"][positions]):
                raise ValueError(f"{split} aligned {field} differs from frozen feature source")
            if array_sha256(value) != manifest["identity_array_hashes"][split][field]:
                raise ValueError(f"{split} aligned {field} hash mismatch")
        for branch in BRANCH_ORDER:
            value = np.asarray(aligned[f"{split}_{branch}"])
            if value.shape != (count, BRANCH_DIMS[branch]) or value.dtype != np.float32:
                raise ValueError(f"{split} {branch} shape or dtype mismatch")
            if not np.isfinite(value).all():
                raise ValueError(f"{split} {branch} contains non-finite values")
            if not np.array_equal(value, source[f"{split}_{branch}"][positions]):
                raise ValueError(f"{split} {branch} is not byte-identical to frozen source")
            if array_sha256(value) != manifest["branch_hashes"][split][branch]:
                raise ValueError(f"{split} {branch} hash mismatch")
    return {
        "valid": True,
        "walk": int(manifest["walk"]),
        "train_rows": int(manifest["row_counts"]["train"]),
        "test_rows": int(manifest["row_counts"]["test"]),
        "concat_dim": 445,
        "all_features_byte_identical_to_frozen_source": True,
        "target_independent": True,
        "feature_bundle_sha256": manifest["feature_bundle_sha256"],
    }
