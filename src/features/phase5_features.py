"""Phase 5 five-branch frozen feature extraction and replay validation."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from data_processing.phase5_walks import validate_phase5_bundle_files
from features.statistical import compute_statistical_features, statistical_feature_dim
from features.transform import compute_transform_features, transform_feature_dim
from training.phase5_encoder import (
    Phase5EncoderConfig,
    build_model,
    sha256_file,
    validate_encoder_run,
    write_json,
)


BRANCH_ORDER = ("statistical", "transformed", "vae", "contrastive", "byol")
BRANCH_DIMS = {
    "statistical": statistical_feature_dim(n_cols=5, ar_order=5),
    "transformed": transform_feature_dim(n_cols=5),
    "vae": 64,
    "contrastive": 128,
    "byol": 128,
}
IDENTITY_FIELDS = (
    "condition_ids",
    "window_start_ns",
    "decision_date_ns",
    "decision_availability_ns",
)
FEATURE_SCHEMA_VERSION = "phase5-feature-v1"


def array_sha256(array: np.ndarray) -> str:
    """Hash dtype, shape, and C-order bytes so array provenance is unambiguous."""

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("utf-8"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _deterministic_chunk(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    statistical = np.stack(
        [compute_statistical_features(sequence) for sequence in sequences]
    ).astype(np.float32, copy=False)
    transformed = np.stack(
        [compute_transform_features(sequence) for sequence in sequences]
    ).astype(np.float32, copy=False)
    return statistical, transformed


def _chunk_ranges(count: int, chunk_size: int) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [(start, min(start + chunk_size, count)) for start in range(0, count, chunk_size)]


def _load_valid_chunk(path: Path, start: int, end: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as chunk:
            statistical = np.asarray(chunk["statistical"], dtype=np.float32)
            transformed = np.asarray(chunk["transformed"], dtype=np.float32)
            stored_start = int(chunk["start"])
            stored_end = int(chunk["end"])
    except (OSError, ValueError, KeyError):
        return None
    expected_rows = end - start
    if (
        stored_start != start
        or stored_end != end
        or statistical.shape != (expected_rows, BRANCH_DIMS["statistical"])
        or transformed.shape != (expected_rows, BRANCH_DIMS["transformed"])
        or not np.isfinite(statistical).all()
        or not np.isfinite(transformed).all()
    ):
        return None
    return statistical, transformed


def extract_deterministic_features(
    sequences: np.ndarray,
    work_dir: Path,
    split: str,
    *,
    chunk_size: int = 256,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract deterministic branches with restartable, validated chunks."""

    values = np.asarray(sequences, dtype=np.float32)
    ranges = _chunk_ranges(len(values), chunk_size)
    work_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[int, int, Path]] = []
    results: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for start, end in ranges:
        path = work_dir / f"{split}_{start:06d}_{end:06d}.npz"
        cached = _load_valid_chunk(path, start, end)
        if cached is None:
            pending.append((start, end, path))
        else:
            results[(start, end)] = cached

    def store(start: int, end: int, path: Path, result: tuple[np.ndarray, np.ndarray]) -> None:
        statistical, transformed = result
        if not np.isfinite(statistical).all() or not np.isfinite(transformed).all():
            raise FloatingPointError(f"non-finite deterministic feature in {split}[{start}:{end}]")
        _atomic_savez(
            path,
            {
                "start": np.asarray(start, dtype=np.int64),
                "end": np.asarray(end, dtype=np.int64),
                "statistical": statistical,
                "transformed": transformed,
            },
        )
        results[(start, end)] = (statistical, transformed)
        print(f"features split={split} rows={start}:{end} complete", flush=True)

    if workers <= 1:
        for start, end, path in pending:
            store(start, end, path, _deterministic_chunk(values[start:end]))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                (start, end, path, executor.submit(_deterministic_chunk, values[start:end]))
                for start, end, path in pending
            ]
            for start, end, path, future in futures:
                store(start, end, path, future.result())

    statistical = np.concatenate([results[key][0] for key in ranges], axis=0)
    transformed = np.concatenate([results[key][1] for key in ranges], axis=0)
    return statistical, transformed


def _config_from_checkpoint(checkpoint: Mapping[str, Any]) -> Phase5EncoderConfig:
    payload = dict(checkpoint["training_config"])
    payload["snapshot_epochs"] = tuple(payload["snapshot_epochs"])
    return Phase5EncoderConfig(**payload)


@torch.no_grad()
def extract_neural_features(
    sequences: np.ndarray,
    checkpoint_path: Path,
    encoder: str,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("phase") != 5
        or checkpoint.get("purpose") != "encoder_pretraining"
        or checkpoint.get("encoder") != encoder
        or int(checkpoint.get("completed_epoch", -1)) != 50
    ):
        raise ValueError(f"checkpoint is not the canonical Phase 5 epoch-50 {encoder} model")
    config = _config_from_checkpoint(checkpoint)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    output = []
    tensor = torch.from_numpy(np.asarray(sequences, dtype=np.float32))
    for batch in tensor.split(batch_size):
        batch = batch.to(device)
        if encoder == "vae":
            embedding = model.encode(batch)[0]
        elif encoder == "contrastive":
            embedding = model(batch)[0]
        else:
            embedding = model.encode(batch)
        output.append(embedding.cpu().numpy().astype(np.float32, copy=False))
    result = np.concatenate(output, axis=0)
    if result.shape != (len(sequences), BRANCH_DIMS[encoder]) or not np.isfinite(result).all():
        raise ValueError(f"invalid extracted {encoder} feature array")
    return result


def _identity_tuples(bundle: Mapping[str, np.ndarray], split: str) -> list[tuple[Any, ...]]:
    arrays = [np.asarray(bundle[f"{split}_{field}"]) for field in IDENTITY_FIELDS]
    return list(zip(*(array.tolist() for array in arrays)))


def audit_supervised_encoder_alignment(dataset_path: Path) -> dict[str, Any]:
    """Fail closed on the encoder/supervised context and identity relationship."""

    with np.load(dataset_path, allow_pickle=False) as source:
        encoder_identities = _identity_tuples(source, "encoder_train")
        train_identities = _identity_tuples(source, "train")
        test_identities = _identity_tuples(source, "test")
        if len(set(encoder_identities)) != len(encoder_identities):
            raise ValueError("encoder identities are not unique")
        if len(set(train_identities)) != len(train_identities):
            raise ValueError("supervised training identities are not unique")
        if len(set(test_identities)) != len(test_identities):
            raise ValueError("evaluation identities are not unique")
        encoder_lookup = {identity: index for index, identity in enumerate(encoder_identities)}
        try:
            mapped = np.asarray([encoder_lookup[identity] for identity in train_identities])
        except KeyError as error:
            raise ValueError("supervised training identity is absent from encoder training") from error
        if set(test_identities) & set(encoder_identities):
            raise ValueError("evaluation identities overlap encoder training")
        if set(test_identities) & set(train_identities):
            raise ValueError("evaluation identities overlap supervised training")
        paired_fields = (
            "sequences",
            "raw_sequences",
            "context_is_imputed",
            "context_is_observed",
            "context_original_gap_length_bars",
            "context_time_since_observation",
            "context_imputed_rows",
            "activity_change_count_24h",
            "lifecycle_fraction",
            "lifecycle_stage",
        )
        for field in paired_fields:
            encoder_values = np.asarray(source[f"encoder_train_{field}"])[mapped]
            train_values = np.asarray(source[f"train_{field}"])
            if not np.array_equal(encoder_values, train_values):
                raise ValueError(f"mapped encoder/train {field} is not byte-identical")
    return {
        "valid": True,
        "encoder_rows": len(encoder_identities),
        "supervised_train_rows": len(train_identities),
        "evaluation_rows": len(test_identities),
        "mapped_train_contexts_byte_identical": True,
        "evaluation_disjoint": True,
    }


def _checkpoint_paths(encoder_root: Path, walk: int) -> dict[str, Path]:
    return {
        encoder: encoder_root / f"walk{walk}" / encoder / "seed0" / "e50" / "checkpoint.pth"
        for encoder in ("vae", "contrastive", "byol")
    }


def build_phase5_feature_bundle(
    dataset_path: Path,
    encoder_root: Path,
    output_path: Path,
    *,
    walk: int,
    encoder_dataset_path: Path | None = None,
    reuse_feature_path: Path | None = None,
    reuse_dataset_path: Path | None = None,
    device: str = "cuda",
    batch_size: int = 1024,
    deterministic_chunk_size: int = 256,
    workers: int = 1,
) -> dict[str, Any]:
    if output_path.exists() or Path(f"{output_path}.manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite Phase 5 feature bundle: {output_path}")
    data_validation = validate_phase5_bundle_files(dataset_path)
    if int(data_validation["walk"]) != walk:
        raise ValueError("source bundle walk mismatch")
    alignment = audit_supervised_encoder_alignment(dataset_path)
    encoder_dataset_path = Path(encoder_dataset_path or dataset_path)
    validate_phase5_bundle_files(encoder_dataset_path)
    with np.load(dataset_path, allow_pickle=False) as downstream_source, np.load(
        encoder_dataset_path, allow_pickle=False
    ) as encoder_source:
        for key in (
            "encoder_train_condition_ids",
            "encoder_train_window_start_ns",
            "encoder_train_decision_date_ns",
            "encoder_train_decision_availability_ns",
            "encoder_train_sequences",
            "encoder_train_raw_sequences",
        ):
            if not np.array_equal(downstream_source[key], encoder_source[key]):
                raise ValueError(
                    f"downstream and checkpoint-source encoder populations differ: {key}"
                )
    checkpoint_paths = _checkpoint_paths(encoder_root, walk)
    checkpoint_validation = {}
    for encoder, checkpoint_path in checkpoint_paths.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_validation[encoder] = validate_encoder_run(
            encoder_dataset_path, checkpoint_path.parents[1]
        )

    reuse_features: dict[str, np.ndarray] | None = None
    reuse_source: dict[str, np.ndarray] | None = None
    reuse_manifest: dict[str, Any] | None = None
    if reuse_feature_path is not None or reuse_dataset_path is not None:
        if reuse_feature_path is None or reuse_dataset_path is None:
            raise ValueError("reuse_feature_path and reuse_dataset_path must be provided together")
        validate_phase5_feature_bundle(reuse_feature_path, dataset_path=reuse_dataset_path)
        reuse_manifest = json.loads(
            Path(f"{reuse_feature_path}.manifest.json").read_text(encoding="utf-8")
        )
        for encoder, checkpoint_path in checkpoint_paths.items():
            if reuse_manifest["checkpoint_provenance"][encoder]["sha256"] != sha256_file(
                checkpoint_path
            ):
                raise ValueError(f"reused {encoder} features came from a different checkpoint")
        with np.load(reuse_feature_path, allow_pickle=False) as stored:
            reuse_features = {name: np.asarray(stored[name]) for name in stored.files}
        with np.load(reuse_dataset_path, allow_pickle=False) as stored:
            reuse_source = {name: np.asarray(stored[name]) for name in stored.files}

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for feature extraction but is unavailable")
    with np.load(dataset_path, allow_pickle=False) as source:
        sequences = {
            split: np.asarray(source[f"{split}_sequences"], dtype=np.float32)
            for split in ("train", "test")
        }
        identities = {
            f"{split}_{field}": np.asarray(source[f"{split}_{field}"])
            for split in ("train", "test")
            for field in IDENTITY_FIELDS
        }

    arrays: dict[str, np.ndarray] = dict(identities)
    work_dir = output_path.parent / f".{output_path.stem}.work"
    reuse_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        missing = np.arange(len(sequences[split]), dtype=np.int64)
        positions = np.full(len(sequences[split]), -1, dtype=np.int64)
        if reuse_features is not None and reuse_source is not None:
            reuse_identities = _identity_tuples(reuse_source, split)
            lookup = {identity: index for index, identity in enumerate(reuse_identities)}
            current_identities = _identity_tuples(identities, split)
            positions = np.asarray([lookup.get(identity, -1) for identity in current_identities])
            matched = np.flatnonzero(positions >= 0)
            missing = np.flatnonzero(positions < 0)
            if not np.array_equal(
                sequences[split][matched],
                reuse_source[f"{split}_sequences"][positions[matched]],
            ):
                raise ValueError(f"reused {split} feature identities do not have identical contexts")
            for branch in BRANCH_ORDER:
                target = np.empty((len(sequences[split]), BRANCH_DIMS[branch]), dtype=np.float32)
                target[matched] = reuse_features[f"{split}_{branch}"][positions[matched]]
                arrays[f"{split}_{branch}"] = target
            reuse_counts[split] = {
                "reused": int(len(matched)),
                "computed": int(len(missing)),
            }
        else:
            reuse_counts[split] = {"reused": 0, "computed": int(len(missing))}
            for branch in BRANCH_ORDER:
                arrays[f"{split}_{branch}"] = np.empty(
                    (len(sequences[split]), BRANCH_DIMS[branch]), dtype=np.float32
                )

        if len(missing):
            statistical, transformed = extract_deterministic_features(
                sequences[split][missing],
                work_dir,
                f"{split}_new",
                chunk_size=deterministic_chunk_size,
                workers=workers,
            )
            arrays[f"{split}_statistical"][missing] = statistical
            arrays[f"{split}_transformed"][missing] = transformed
            for encoder, checkpoint_path in checkpoint_paths.items():
                print(
                    f"features walk={walk} split={split} encoder={encoder} "
                    f"new_rows={len(missing)} starting",
                    flush=True,
                )
                arrays[f"{split}_{encoder}"][missing] = extract_neural_features(
                    sequences[split][missing],
                    checkpoint_path,
                    encoder,
                    device=resolved_device,
                    batch_size=batch_size,
                )
    branch_hashes = {
        split: {
            branch: array_sha256(arrays[f"{split}_{branch}"])
            for branch in BRANCH_ORDER
        }
        for split in ("train", "test")
    }
    identity_hashes = {
        split: {
            field: array_sha256(arrays[f"{split}_{field}"])
            for field in IDENTITY_FIELDS
        }
        for split in ("train", "test")
    }
    source_manifest = json.loads(Path(f"{dataset_path}.manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "phase": 5,
        "purpose": "frozen_five_branch_features",
        "walk": walk,
        "source_dataset_path": str(dataset_path),
        "source_dataset_sha256": sha256_file(dataset_path),
        "source_manifest_sha256": sha256_file(Path(f"{dataset_path}.manifest.json")),
        "encoder_dataset_path": str(encoder_dataset_path),
        "encoder_dataset_sha256": sha256_file(encoder_dataset_path),
        "encoder_population_matches_checkpoint_source": True,
        "source_identity_hashes": {
            split: source_manifest["identity_hashes"][split] for split in ("train", "test")
        },
        "row_counts": {split: int(len(sequences[split])) for split in ("train", "test")},
        "branch_order": list(BRANCH_ORDER),
        "branch_dims": BRANCH_DIMS,
        "concat_dim": int(sum(BRANCH_DIMS.values())),
        "branch_hashes": branch_hashes,
        "identity_array_hashes": identity_hashes,
        "checkpoint_provenance": {
            encoder: {
                "path": str(path),
                "sha256": sha256_file(path),
                "completed_epoch": 50,
                "validation": checkpoint_validation[encoder],
            }
            for encoder, path in checkpoint_paths.items()
        },
        "feature_reuse": {
            "source_feature_path": str(reuse_feature_path) if reuse_feature_path else None,
            "source_feature_sha256": sha256_file(reuse_feature_path) if reuse_feature_path else None,
            "source_dataset_path": str(reuse_dataset_path) if reuse_dataset_path else None,
            "source_dataset_sha256": sha256_file(reuse_dataset_path) if reuse_dataset_path else None,
            "counts": reuse_counts,
            "identity_and_context_match_required": True,
        },
        "alignment_audit": alignment,
        "fitted_on_evaluation": False,
        "input_key": {"train": "train_sequences", "test": "test_sequences"},
    }
    _atomic_savez(output_path, arrays)
    manifest["feature_bundle_sha256"] = sha256_file(output_path)
    write_json(Path(f"{output_path}.manifest.json"), manifest)
    return validate_phase5_feature_bundle(output_path, dataset_path=dataset_path)


def validate_phase5_feature_bundle(
    feature_path: Path,
    *,
    dataset_path: Path | None = None,
    replay_sample_size: int = 8,
) -> dict[str, Any]:
    manifest_path = Path(f"{feature_path}.manifest.json")
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("feature bundle or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 5 feature schema")
    if manifest.get("branch_order") != list(BRANCH_ORDER) or manifest.get("branch_dims") != BRANCH_DIMS:
        raise ValueError("feature branch contract mismatch")
    if int(manifest.get("concat_dim", -1)) != 445:
        raise ValueError("Phase 5 concat feature dimension must be 445")
    if sha256_file(feature_path) != manifest.get("feature_bundle_sha256"):
        raise ValueError("feature bundle file hash mismatch")
    source_path = Path(dataset_path or manifest["source_dataset_path"])
    validate_phase5_bundle_files(source_path)
    if sha256_file(source_path) != manifest.get("source_dataset_sha256"):
        raise ValueError("feature source dataset hash mismatch")
    if sha256_file(Path(f"{source_path}.manifest.json")) != manifest.get("source_manifest_sha256"):
        raise ValueError("feature source manifest hash mismatch")
    encoder_dataset_path = Path(manifest.get("encoder_dataset_path", source_path))
    if not encoder_dataset_path.is_file():
        raise FileNotFoundError("feature encoder-source dataset is missing")
    if sha256_file(encoder_dataset_path) != manifest.get(
        "encoder_dataset_sha256", manifest.get("source_dataset_sha256")
    ):
        raise ValueError("feature encoder-source dataset hash mismatch")
    if manifest.get("encoder_population_matches_checkpoint_source", True) is not True:
        raise ValueError("feature manifest does not affirm encoder-source alignment")
    reuse = manifest.get("feature_reuse", {})
    if reuse.get("source_feature_path") is not None:
        reuse_feature = Path(reuse["source_feature_path"])
        reuse_dataset = Path(reuse["source_dataset_path"])
        if not reuse_feature.is_file() or not reuse_dataset.is_file():
            raise FileNotFoundError("feature reuse source is missing")
        if sha256_file(reuse_feature) != reuse.get("source_feature_sha256"):
            raise ValueError("feature reuse source hash mismatch")
        if sha256_file(reuse_dataset) != reuse.get("source_dataset_sha256"):
            raise ValueError("feature reuse dataset hash mismatch")
        if reuse.get("identity_and_context_match_required") is not True:
            raise ValueError("feature reuse did not require identity/context matching")

    with np.load(feature_path, allow_pickle=False) as features, np.load(
        source_path, allow_pickle=False
    ) as source:
        for split in ("train", "test"):
            count = int(manifest["row_counts"][split])
            for field in IDENTITY_FIELDS:
                key = f"{split}_{field}"
                if not np.array_equal(features[key], source[key]):
                    raise ValueError(f"feature/source identity mismatch for {key}")
                if array_sha256(features[key]) != manifest["identity_array_hashes"][split][field]:
                    raise ValueError(f"feature identity hash mismatch for {key}")
            for branch in BRANCH_ORDER:
                key = f"{split}_{branch}"
                value = np.asarray(features[key])
                if value.shape != (count, BRANCH_DIMS[branch]) or value.dtype != np.float32:
                    raise ValueError(f"invalid feature shape/dtype for {key}")
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite values in {key}")
                if array_sha256(value) != manifest["branch_hashes"][split][branch]:
                    raise ValueError(f"feature hash mismatch for {key}")
        sample_count = min(replay_sample_size, int(manifest["row_counts"]["train"]))
        for index in range(sample_count):
            sequence = source["train_sequences"][index]
            expected_statistical = compute_statistical_features(sequence)
            expected_transformed = compute_transform_features(sequence)
            if not np.array_equal(expected_statistical, features["train_statistical"][index]):
                raise ValueError("statistical feature sample replay mismatch")
            if not np.array_equal(expected_transformed, features["train_transformed"][index]):
                raise ValueError("transformed feature sample replay mismatch")
    alignment = audit_supervised_encoder_alignment(source_path)
    return {
        "valid": True,
        "walk": int(manifest["walk"]),
        "train_rows": int(manifest["row_counts"]["train"]),
        "test_rows": int(manifest["row_counts"]["test"]),
        "concat_dim": 445,
        "feature_bundle_sha256": manifest["feature_bundle_sha256"],
        "alignment_audit": alignment,
    }


def load_concat_features(feature_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load canonical branches in their frozen order and concatenate them."""

    with np.load(feature_path, allow_pickle=False) as bundle:
        train = np.concatenate([bundle[f"train_{branch}"] for branch in BRANCH_ORDER], axis=1)
        test = np.concatenate([bundle[f"test_{branch}"] for branch in BRANCH_ORDER], axis=1)
    if train.shape[1] != 445 or test.shape[1] != 445:
        raise ValueError("invalid canonical concat feature dimension")
    return train.astype(np.float32, copy=False), test.astype(np.float32, copy=False)
