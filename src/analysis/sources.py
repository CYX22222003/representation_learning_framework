"""Read-only source adapters with exact row, target and lineage verification."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np

from .artifacts import IDENTITY_FIELDS, hash_array, hash_file, read_json
from .contracts import BRANCHES
from .runtime import resolve_recorded_path
from .types import (AnalysisContext, PredictionRecord, SnapshotEpoch, AnalysisTask,
                    RowIdentity, Split, TaskDataset, TaskSplit, TaskWalkSource, ValidationResult)


def locate_source(context: AnalysisContext, task: AnalysisTask, walk: int) -> TaskWalkSource:
    task = AnalysisTask(task)
    if walk not in (1, 2):
        raise ValueError("Unknown calendar walk")
    root = context.repo_root
    if task == AnalysisTask.CLASSIFICATION:
        base = root / "experiments/phase5"
        data = base / f"data_preparation/walk{walk}/market_1h_seq64_h2.npz"
        feature = base / f"features/walk{walk}/five_branch_epoch50.npz"
        h0 = base / f"downstream/walk{walk}/classification/seed0"
        scaler = base / f"downstream/walk{walk}/feature_standardizer.npz"
    elif task == AnalysisTask.PRICE:
        base = root / "experiments/phase5/downstream_addons"
        data = base / f"shared/h8/data/walk{walk}/market_1h_seq64_h8.npz"
        feature = base / f"shared/h8/features/walk{walk}/five_branch_epoch50.npz"
        h0 = base / f"tasks/absolute_price_h8/walk{walk}/seed0"
        scaler = base / f"shared/h8/feature_scalers/walk{walk}/feature_standardizer.npz"
    else:
        base = root / "experiments/phase6/volatility_prediction"
        data = base / f"data_preparation/walk{walk}/volatility_1h_seq64_h8.npz"
        feature = base / f"features/walk{walk}/five_branch_epoch50_h8.npz"
        h0 = base / f"runs/framework_h0/walk{walk}/seed0"
        scaler = h0 / "feature_standardizer.npz"
    from .profiles import adapt_source
    return adapt_source(TaskWalkSource(task, walk, data, feature, Path(f"{data}.manifest.json"),
                          Path(f"{feature}.manifest.json"), h0, scaler), context)


def readonly(a):
    result = np.asarray(a)
    result.setflags(write=False)
    return result


def validate_identity(identity: RowIdentity) -> None:
    n = len(identity.condition_ids)
    arrays = [np.asarray(getattr(identity, f)) for f in IDENTITY_FIELDS]
    if any(a.ndim != 1 or len(a) != n for a in arrays):
        raise ValueError("Identity columns must be aligned vectors")
    if any(len(a) != n for a in identity.target_fields.values()):
        raise ValueError("Target identity columns are misaligned")
    keys = list(zip(*(a.tolist() for a in arrays)))
    if len(set(keys)) != n:
        raise ValueError("Duplicate row identities")
    decision = identity.decision_date_ns
    if np.any(identity.window_start_ns > decision) or np.any(identity.decision_availability_ns < decision):
        raise ValueError("Invalid decision/context chronology")


def load_dataset(source: TaskWalkSource, context: AnalysisContext) -> TaskDataset:
    if source.profile and source.profile.family != "canonical":
        from .profile_sources import load_selected_dataset
        return load_selected_dataset(source, context)
    dm, fm = read_json(source.dataset_manifest_path), read_json(source.feature_manifest_path)
    for manifest in (dm, fm):
        if manifest["walk"] != source.walk:
            raise ValueError("Source manifest walk mismatch")
    expected = dm.get("artifact", {}).get("sha256")
    if not expected or hash_file(source.dataset_path) != expected:
        raise ValueError("Dataset file hash mismatch")
    if hash_file(source.feature_path) != fm["feature_bundle_sha256"]:
        raise ValueError("Feature file hash mismatch")
    if fm["branch_order"] != [b.key for b in BRANCHES] or fm["branch_dims"] != {b.key: b.width for b in BRANCHES}:
        raise ValueError("Canonical branch contract mismatch")
    if fm.get("fitted_on_evaluation", False):
        raise ValueError("Source features were fitted on evaluation data")
    splits = []
    with np.load(source.dataset_path, allow_pickle=False) as data, np.load(source.feature_path, allow_pickle=False) as features:
        for split in Split:
            prefix = split.value + "_"
            target_fields = {k[len(prefix):]: readonly(data[k]) for k in data.files
                             if k.startswith(prefix + "target_") and
                             (k.endswith("_ns") or k.endswith("_is_observed") or k.endswith("_is_imputed"))}
            identity = RowIdentity(source.task, source.walk,
                *(readonly(data[prefix + f]) for f in IDENTITY_FIELDS), MappingProxyType(target_fields))
            validate_identity(identity)
            for field in IDENTITY_FIELDS:
                value = features[prefix + field]
                if not np.array_equal(value, getattr(identity, field)):
                    raise ValueError(f"Feature/data identity mismatch: {split.value}/{field}")
                if hash_array(value) != fm["identity_array_hashes"][split.value][field]:
                    raise ValueError("Identity array hash mismatch")
            target_key = {AnalysisTask.CLASSIFICATION: "classification_labels",
                          AnalysisTask.PRICE: "target_close",
                          AnalysisTask.VARIANCE: "realised_variance"}[source.task]
            targets = readonly(data[prefix + target_key])
            if targets.ndim != 1 or not np.isfinite(targets).all():
                raise ValueError("Task targets must be finite vectors")
            if source.task == AnalysisTask.CLASSIFICATION:
                delta = data[prefix + "target_close"] - data[prefix + "current_close"]
                expected_labels = np.where(delta < -0.001, 0, np.where(delta > 0.001, 2, 1))
                if not np.array_equal(targets, expected_labels):
                    raise ValueError("Classification target contract mismatch")
            if source.task == AnalysisTask.VARIANCE:
                closes = data[prefix + "target_close_path"]
                if closes.shape != (len(targets), 9) or not np.array_equal(
                    np.square(np.diff(closes, axis=1)).sum(axis=1), targets):
                    raise ValueError("RV future-path target mismatch")
                if not data[prefix + "target_path_is_observed"].all() or data[prefix + "target_path_is_imputed"].any():
                    raise ValueError("RV target path is not fully observed")
            branches = {}
            for branch in BRANCHES:
                value = readonly(features[prefix + branch.key])
                if value.shape != (len(targets), branch.width) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"Invalid feature block: {branch.key}")
                if hash_array(value) != fm["branch_hashes"][split.value][branch.key]:
                    raise ValueError(f"Feature array hash mismatch: {branch.key}")
                branches[branch.key] = value
            metadata = {}
            for key in ("current_close", "target_close", "target_date_ns", "target_end_ns",
                        "target_availability_ns", "raw_sequences"):
                if prefix + key in data:
                    metadata[key] = readonly(data[prefix + key])
            if "raw_sequences" in metadata:
                closes = metadata.pop("raw_sequences")[:, :, 3].astype(np.float64)
                metadata.update(last_hour_reversal=readonly(-(closes[:, -1] - closes[:, -2])),
                                historical_rv=readonly(np.square(np.diff(closes[:, -9:], axis=1)).sum(axis=1)))
            intervals = dm["intervals"]
            training = intervals["training"]
            cutoff = training.get("cutoff_exclusive", training.get("cutoff_exclusive_for_target_availability"))
            end = intervals["evaluation"].get("end_exclusive", intervals["evaluation"].get("end_inclusive_for_target_availability"))
            boundary = int(np.datetime64(cutoff.replace("+00:00", ""), "ns").astype(np.int64))
            eval_end = int(np.datetime64(end.replace("+00:00", ""), "ns").astype(np.int64))
            availability = data[prefix + "target_availability_ns"]
            if split == Split.TRAIN and np.any(availability >= boundary):
                raise ValueError("Training target crosses global cutoff")
            if split == Split.TEST and (np.any(identity.decision_availability_ns < boundary) or np.any(availability > eval_end)):
                raise ValueError("Evaluation rows cross walk interval")
            if "target_date_ns" in metadata:
                horizon = 2 if source.task == AnalysisTask.CLASSIFICATION else 8
                if not np.array_equal(metadata["target_date_ns"], identity.decision_date_ns + horizon * 3600 * 10**9):
                    raise ValueError("Target horizon mismatch")
            splits.append(TaskSplit(split, identity, targets, MappingProxyType(branches), MappingProxyType(metadata)))
    train_keys = set(zip(splits[0].identity.condition_ids.tolist(), splits[0].identity.decision_date_ns.tolist()))
    test_keys = set(zip(splits[1].identity.condition_ids.tolist(), splits[1].identity.decision_date_ns.tolist()))
    if train_keys & test_keys:
        raise ValueError("Training and evaluation identities overlap")
    return TaskDataset(source, *splits, MappingProxyType({
        "dataset": hash_file(source.dataset_path), "features": hash_file(source.feature_path),
        "dataset_manifest": hash_file(source.dataset_manifest_path), "feature_manifest": hash_file(source.feature_manifest_path)}))


def audit_source(source: TaskWalkSource, context: AnalysisContext) -> ValidationResult:
    from .runtime import inventory_dependencies
    try:
        inventory = inventory_dependencies((source,), context)
        manifest = read_json(source.dataset_manifest_path)
        with np.load(source.dataset_path, allow_pickle=False) as stored:
            arrays = {key: np.asarray(stored[key]) for key in stored.files}
        if source.task == AnalysisTask.VARIANCE:
            from data_processing.phase6_volatility_labels import validate_volatility_label_arrays
            validate_volatility_label_arrays(arrays, manifest)
        else:
            from data_processing.phase5_walks import validate_phase5_arrays
            validate_phase5_arrays(arrays, manifest)
        del arrays
        dataset = load_dataset(source, context)
        return ValidationResult(f"{source.task.value}/walk{source.walk}", "valid",
            {"dependencies": str(len(inventory)), "train_rows": str(len(dataset.train.targets)),
             "test_rows": str(len(dataset.test.targets))})
    except (ValueError, KeyError, OSError) as exc:
        return ValidationResult(f"{source.task.value}/walk{source.walk}", "invalid", {"failure": str(exc)})


def load_h0_predictions(source: TaskWalkSource, epoch: SnapshotEpoch, context: AnalysisContext,
                        dataset: TaskDataset | None = None) -> PredictionRecord:
    dataset = dataset or load_dataset(source, context)
    path = source.h0_run_path / f"e{epoch}" / "predictions.npz"
    metrics = read_json(path.parent / "metrics.json")
    if hash_file(path) != metrics["predictions_sha256"]:
        raise ValueError("Original H0 prediction hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        for field in IDENTITY_FIELDS:
            if not np.array_equal(saved[field], getattr(dataset.test.identity, field)):
                raise ValueError(f"H0 identity mismatch: {field}")
        if source.task == AnalysisTask.CLASSIFICATION:
            outputs, targets, units = saved["logits"], saved["targets"], "class_logits"
        elif source.task == AnalysisTask.PRICE:
            outputs, targets, units = saved["prediction_future_price"], saved["target_future_price"], "raw_probability"
        else:
            outputs, targets, units = saved["prediction_realised_variance"], saved["target_realised_variance"], "raw_realised_variance"
        if not np.array_equal(targets, dataset.test.targets):
            raise ValueError("H0 target mismatch")
        for key, value in dataset.test.identity.target_fields.items():
            if key in saved and not np.array_equal(saved[key], value):
                raise ValueError(f"H0 target identity mismatch: {key}")
        outputs = readonly(outputs)
    from .profiles import reference_name
    return PredictionRecord(source.task, source.walk, reference_name(source), 0, epoch, dataset.test.identity,
        dataset.test.targets, outputs, units, {**dataset.source_hashes, "checkpoint": metrics["checkpoint_sha256"]},
        dataset.test.metadata)
