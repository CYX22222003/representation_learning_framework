"""Tolerance-aware replay with visible warnings and isolated invalid results."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import (hash_file, identity_hash, json_value, load_prediction, read_json,
                        snapshot_path)
from .contracts import RECIPE
from .profiles import reference_name, similarity_sources, source_branches
from .features import (apply_standardizer, fit_standardizer, load_standardizer,
                       select_features, validate_scaler_parity)
from .sources import load_dataset, load_h0_predictions
from .tasks import evaluate_predictions
from .types import (AblationKind, AblationSpec, AnalysisContext, ExperimentMatrix,
                    ProbeRunSpec, SimilarityResult, SnapshotEpoch, TaskDataset,
                    TaskWalkSource, ValidationObservation, ValidationResult)
from .validation import compare_numbers, emit_validation, failure, merge_results, result


def output_atol(task, policy):
    return {"classification_h2": policy.logit_atol, "absolute_price_h8": policy.probability_atol,
            "realised_variance": policy.variance_atol}[task.value]


def metric_atol(task, name, policy):
    if name == "rank_ic_median" and policy.rank_median_metric_atol is not None:
        return policy.rank_median_metric_atol
    if name in ("spearman", "implied_spearman", "rank_ic", "rank_ic_median"):
        return policy.rank_metric_atol
    return policy.classification_metric_atol if task == "classification_h2" else policy.metric_atol


def compare_metric_rows(expected, saved, policy):
    if len(expected) != len(saved):
        raise ValueError("Metric row inventory changed")
    observations = []
    for left, right in zip(expected, saved):
        for key in ("task", "walk", "configuration", "epoch", "group", "diagnostics"):
            if left.get(key) != right.get(key):
                raise ValueError(f"Metric row definition changed: {key}")
        if len(left["metrics"]) != len(right["metrics"]):
            raise ValueError("Metric inventory changed")
        for actual, recorded in zip(left["metrics"], right["metrics"]):
            for key in ("name", "direction", "count", "availability", "reason"):
                if actual[key] != recorded[key]:
                    raise ValueError(f"Metric definition changed: {key}")
            if actual["value"] is None or recorded["value"] is None:
                if actual["value"] != recorded["value"]:
                    raise ValueError("Metric availability changed")
                continue
            name = actual["name"]
            atol = metric_atol(left["task"], name, policy)
            observations.append(compare_numbers(f"{left['group']}:{name}", actual["value"], recorded["value"],
                                rtol=policy.rtol, atol=atol))
    return observations


def replay_metrics(record, actual, policy):
    expected = {m.name: m for m in evaluate_predictions(record)}
    observed = {m.name: m for m in evaluate_predictions(actual)}
    checks = []
    for name, metric in expected.items():
        other = observed[name]
        if metric.value is None or other.value is None:
            if metric.value != other.value:
                checks.append(ValidationObservation(name, "error", "Metric availability changed"))
            continue
        atol = metric_atol(record.task.value, name, policy)
        checks.append(compare_numbers("metric:" + name, other.value, metric.value, rtol=policy.rtol, atol=atol))
    if record.task.value == "classification_h2":
        changed = int(np.sum(record.outputs.argmax(axis=1) != actual.outputs.argmax(axis=1)))
        if changed:
            checks.append(ValidationObservation("class_decisions", "warning",
                         "Near-tied class decisions changed during numerical replay", {"changed_rows": changed}))
    return checks


def validate_preparation(context: AnalysisContext, matrix: ExperimentMatrix) -> ValidationResult:
    validations = []
    for source in matrix.references:
        try:
            dataset = load_dataset(source, context)
            canonical = load_standardizer(source.h0_scaler_path)
            for spec in (r for r in matrix.runs if r.source == source):
                branches = source_branches(source)
                scaler = fit_standardizer(select_features(dataset.train, spec.ablation, branches), dataset.train.identity)
                validate_scaler_parity(scaler, canonical, spec.ablation, branches)
                path = context.output_root / "features" / source.task.value / f"walk{source.walk}" / spec.ablation.configuration / "feature_standardizer.npz"
                stored = load_standardizer(path)
                if stored.clip_limits != scaler.clip_limits:
                    raise ValueError("Prepared scaler clipping changed")
                for name in ("mean", "raw_std", "scale"):
                    if not np.array_equal(getattr(stored, name), getattr(scaler, name)):
                        raise ValueError("Prepared scaler state changed")
            validations.append(result(f"{source.task.value}/walk{source.walk}"))
        except (ValueError, KeyError, OSError) as exc:
            validations.append(failure(f"{source.task.value}/walk{source.walk}", exc))
    return emit_validation(merge_results("preparation", validations),
                            context.output_root / "manifests/preparation_validation.json")


def replay_h0(source: TaskWalkSource, context: AnalysisContext,
              dataset: TaskDataset | None = None) -> ValidationResult:
    import torch
    from .tasks import build_head
    from .training import predict_output
    try:
        dataset = dataset or load_dataset(source, context)
        manifest = read_json(source.h0_run_path / "dataset_manifest.json")
        if manifest.get("dataset_sha256", manifest.get("label_sha256")) != dataset.source_hashes["dataset"] or manifest["feature_sha256"] != dataset.source_hashes["features"]:
            raise ValueError("H0 source hash mismatch")
        scaler_hash = manifest.get("scaler_sha256", manifest.get("feature_scaler_sha256"))
        if scaler_hash and hash_file(source.h0_scaler_path) != scaler_hash:
            raise ValueError("H0 scaler hash mismatch")
        branches = source_branches(source)
        name = reference_name(source)
        h0 = AblationSpec(name, AblationKind.SINGLE, tuple(b.key for b in branches), sum(b.width for b in branches))
        spec = ProbeRunSpec(source, h0, source.h0_run_path, device="cpu")
        fitted = fit_standardizer(np.concatenate([dataset.train.branches[b.key] for b in branches], axis=1), dataset.train.identity)
        validate_scaler_parity(fitted, load_standardizer(source.h0_scaler_path), h0, branches)
        values = apply_standardizer(np.concatenate([dataset.test.branches[b.key] for b in branches], axis=1),
                                    load_standardizer(source.h0_scaler_path))
        observations = []
        config = read_json(source.h0_run_path / "config.json")
        if source.profile and source.profile.family != "canonical":
            from .artifacts import hash_array
            from data_processing.phase5_walks import sha256_arrays
            architecture = read_json(source.h0_run_path / "architecture_manifest.json")
            if config.get("configuration") != name or config.get("task") != source.task.value or config.get("walk") != source.walk:
                raise ValueError("Selected reference configuration/task/walk mismatch")
            if architecture.get("branches") != list(h0.branches) or architecture.get("input_dim") != h0.width:
                raise ValueError("Selected reference architecture mismatch")
            if manifest.get("configuration") != name or manifest.get("branch_order") != list(h0.branches) or manifest.get("input_dim") != h0.width:
                raise ValueError("Selected reference branch provenance mismatch")
            for key, expected in (("weight_decay", 0.), ("hidden_dim", 128), ("snapshot_epochs", [5, 15, 50])):
                if config.get(key) != expected:
                    raise ValueError(f"Selected reference recipe mismatch: {key}")
            for split in (dataset.train, dataset.test):
                from .artifacts import IDENTITY_FIELDS
                key = split.split.value
                if manifest[key + "_identity_hash"] != sha256_arrays(*(getattr(split.identity, f) for f in IDENTITY_FIELDS)):
                    raise ValueError("Selected reference identity hash mismatch")
                if manifest[key + "_target_hash"] != hash_array(split.targets):
                    raise ValueError("Selected reference target hash mismatch")
            scaler_meta = read_json(source.h0_run_path / "feature_standardizer.manifest.json")
            if scaler_meta["sha256"] != hash_file(source.h0_scaler_path):
                raise ValueError("Selected reference scaler integrity mismatch")
            completion = read_json(source.h0_run_path / "training_complete.json")
            if completion.get("complete") is not True or completion.get("snapshots") != [5, 15, 50] or completion.get("principal_epoch") != 50:
                raise ValueError("Selected reference is incomplete")
        for key, expected in (("seed", 0), ("epochs", 50), ("batch_size", 512), ("learning_rate", 1e-4)):
            if config[key] != expected:
                raise ValueError(f"H0 recipe mismatch: {key}")
        for epoch in RECIPE.snapshots:
            recorded = load_h0_predictions(source, epoch, context, dataset)
            checkpoint_path = source.h0_run_path / f"e{epoch}/checkpoint.pth"
            if hash_file(checkpoint_path) != recorded.hashes["checkpoint"]:
                raise ValueError("H0 checkpoint hash mismatch")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint["completed_epoch"] != epoch:
                raise ValueError("H0 checkpoint epoch mismatch")
            if checkpoint.get("dataset_sha256", checkpoint.get("label_sha256")) != dataset.source_hashes["dataset"] or checkpoint["feature_sha256"] != dataset.source_hashes["features"]:
                raise ValueError("H0 checkpoint source mismatch")
            model = build_head(spec)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            multiplier = 10000. if source.task.value == "realised_variance" else 1.
            outputs = predict_output(model, values, torch.device("cpu"), multiplier)
            outputs = outputs if source.task.value == "classification_h2" else outputs.reshape(-1)
            actual = replace(recorded, outputs=outputs)
            observations.append(compare_numbers(f"e{epoch}:outputs", outputs, recorded.outputs,
                                rtol=context.replay_policy.rtol, atol=output_atol(source.task, context.replay_policy)))
            observations.extend(replay_metrics(recorded, actual, context.replay_policy))
        value = result(f"{name}/{source.task.value}/walk{source.walk}", observations)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        value = failure(f"H0/{source.task.value}/walk{source.walk}", exc)
    return emit_validation(value, context.output_root / "analysis/reference_replay" / source.task.value / f"walk{source.walk}.json")


def replay_snapshot(spec: ProbeRunSpec, dataset: TaskDataset, epoch: SnapshotEpoch,
                    *, context: AnalysisContext | None = None) -> ValidationResult:
    from .training import ProbeRunner, run_manifest
    from .runtime import collect_environment
    if context is None:
        raise ValueError("Replay requires the explicit analysis context and frozen tolerance policy")
    subject = f"{spec.source.task.value}/{spec.ablation.configuration}/walk{spec.source.walk}/e{epoch}"
    diagnostic_path = (spec.output_path / f"e{epoch}/replay_validation.json"
                       if (spec.output_path / "dataset_manifest.json").is_file()
                       else context.output_root / "analysis/replay_failures" / spec.source.task.value /
                       spec.ablation.configuration / f"walk{spec.source.walk}/e{epoch}.json")
    try:
        folder = snapshot_path(spec.output_path, epoch)
        saved = load_prediction(folder / "predictions.npz")
        if identity_hash(saved.identity) != identity_hash(dataset.test.identity) or not np.array_equal(saved.targets, dataset.test.targets):
            raise ValueError("Saved prediction identity/target mismatch")
        if (saved.task, saved.walk, saved.configuration, saved.epoch, saved.seed) != (
            spec.source.task, spec.source.walk, spec.ablation.configuration, epoch, 0):
            raise ValueError("Snapshot configuration metadata mismatch")
        with np.load(folder / "history.npz", allow_pickle=False) as history:
            if not np.array_equal(history["epochs"], np.arange(1, epoch + 1)) or not np.isfinite(history["train_loss"]).all():
                raise ValueError("Snapshot history is incomplete or non-finite")
        meta = read_json(folder / "metrics.json")
        if meta["checkpoint_sha256"] != hash_file(folder / "checkpoint.pth") or meta["predictions_sha256"] != hash_file(folder / "predictions.npz"):
            raise ValueError("Snapshot integrity mismatch")
        runner = ProbeRunner(context, collect_environment(context))
        actual = runner.predict_snapshot(replace(spec, device="cpu"), dataset, epoch)
        observations = [compare_numbers("outputs", actual.outputs, saved.outputs,
                        rtol=context.replay_policy.rtol, atol=output_atol(spec.source.task, context.replay_policy))]
        observations += replay_metrics(saved, actual, context.replay_policy)
        computed = {m.name: m for m in evaluate_predictions(saved)}
        stored_metrics = {m["name"]: m for m in meta["metrics"]}
        if set(computed) != set(stored_metrics):
            raise ValueError("Saved metric inventory changed")
        for name, m in computed.items():
            stored = stored_metrics[name]
            if (stored["count"], stored["direction"], stored["availability"]) != (m.count, m.direction.value, m.availability):
                raise ValueError(f"Saved metric definition/availability changed: {name}")
            if m.value is not None:
                atol = metric_atol(saved.task.value, name, context.replay_policy)
                observations.append(compare_numbers("saved_metric:" + name, m.value, stored["value"],
                                    rtol=context.replay_policy.rtol, atol=atol))
        return emit_validation(result(subject, observations), diagnostic_path)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        return emit_validation(failure(subject, exc), diagnostic_path)


def replay_similarity(value: SimilarityResult, dataset: TaskDataset) -> ValidationResult:
    from .similarity import compute_cka
    try:
        branches = source_branches(dataset.source)
        scaler = fit_standardizer(np.concatenate([dataset.train.branches[b.key] for b in branches], axis=1), dataset.train.identity)
        split = dataset.train if value.sample.split.value == "train" else dataset.test
        actual = compute_cka(split, scaler, value.sample, branches)
        if value.branches != actual.branches or value.unavailable_reasons != actual.unavailable_reasons:
            raise ValueError("CKA branch/availability mismatch")
        valid = np.isfinite(value.scores)
        if not np.array_equal(valid, np.isfinite(actual.scores)):
            raise ValueError("CKA availability changed")
        label = f"CKA/{dataset.source.task.value}" if dataset.source.profile else "CKA"
        return result(f"{label}/walk{dataset.source.walk}/{split.split.value}",
                      (compare_numbers("cka", actual.scores[valid], value.scores[valid], rtol=1e-8, atol=1e-8),))
    except (ValueError, KeyError, OSError) as exc:
        return failure("CKA", exc)


def validate_complete(context: AnalysisContext, matrix: ExperimentMatrix) -> ValidationResult:
    values = []
    pooled_records = []
    for source in matrix.references:
        dataset = None
        try:
            from .sources import audit_source
            audited = emit_validation(audit_source(source, context))
            if not audited.usable:
                raise ValueError("Source lineage failed independent audit")
            dataset = load_dataset(source, context)
            values.append(replay_h0(source, context, dataset))
        except (ValueError, KeyError, OSError) as exc:
            values.append(failure(str(source.dataset_path), exc))
        for spec in (s for s in matrix.runs if s.source == source):
            if dataset is None:
                values.append(failure(str(spec.output_path), ValueError("Invalid shared source")))
                continue
            marker = spec.output_path / "training_complete.json"
            if not marker.is_file():
                values.append(failure(str(spec.output_path), ValueError("Missing training completion")))
                continue
            completion = read_json(marker)
            if completion.get("complete") is not True or completion.get("snapshots") != [5, 15, 50] or completion.get("principal_epoch") != 50:
                values.append(failure(str(marker), ValueError("Invalid completion marker")))
                continue
            checks = [replay_snapshot(spec, dataset, epoch, context=context) for epoch in RECIPE.snapshots]
            summary = emit_validation(merge_results(str(spec.output_path), checks), spec.output_path / "replay_validation.json")
            values.append(summary)
    from .similarity import load_saved_similarity, sample_splits
    structural_sources = similarity_sources(context, matrix)
    for task in dict.fromkeys(s.task for s in structural_sources):
        try:
            populations = [load_dataset(s, context) for s in structural_sources if s.task == task]
            samples = sample_splits(populations)
            for dataset in populations:
                for split in (dataset.train, dataset.test):
                    saved = load_saved_similarity(context, dataset, split.split)
                    expected = next(s for s in samples if (s.walk, s.split) == (dataset.source.walk, split.split))
                    if not np.array_equal(saved.sample.positions, expected.positions):
                        raise ValueError("CKA sample differs from predeclared deterministic sampler")
                    values.append(replay_similarity(saved, dataset))
        except (ValueError, KeyError, OSError) as exc:
            values.append(failure(f"CKA inventory/{task.value}", exc))
    if context.profiles:
        # 7C requires singles/removals/CKA only; conditional tables are optional.
        return emit_validation(merge_results("complete_matrix", values), context.output_root / "manifests/completion_validation.json")
    for source in matrix.references:
        path = context.output_root / "analysis/conditional" / source.task.value / f"walk{source.walk}/results.json"
        try:
            payload = read_json(path)
            if payload["configurations"] != sorted(["H0"] + [r.ablation.configuration for r in matrix.runs if r.source == source]):
                raise ValueError("Incomplete conditional inventory")
            from .conditional import fit_conditions, assign_groups, evaluate_groups, summarize_contracts
            dataset = load_dataset(source, context)
            if payload["source_hashes"] != dict(dataset.source_hashes) or payload["conditions"] != json_value(fit_conditions(dataset)):
                raise ValueError("Conditional source/threshold provenance changed")
            expected_rows = []
            records = [load_h0_predictions(source, 50, context, dataset)]
            records += [load_prediction(r.output_path / "e50/predictions.npz") for r in matrix.runs if r.source == source]
            for record in records:
                grouped = evaluate_groups(record, assign_groups(record, fit_conditions(dataset)))
                expected_rows += list(grouped) + list(summarize_contracts(grouped))
            observations = compare_metric_rows(json_value(expected_rows), payload["rows"], context.replay_policy)
            values.append(result(f"conditional/{source.task.value}/walk{source.walk}", observations))
            pooled_records.extend(records)
        except (ValueError, KeyError, OSError) as exc:
            values.append(failure(str(path), exc))
    pooled = context.output_root / "analysis/conditional/pooled.json"
    if not pooled.is_file():
        values.append(failure(str(pooled), ValueError("Missing pooled metrics")))
    else:
        try:
            from .conditional import pool_walks
            groups = {}
            for record in pooled_records:
                groups.setdefault((record.task, record.configuration), []).append(record)
            expected = []
            for pair in groups.values():
                if len(pair) == 2:
                    expected.extend(pool_walks(pair))
            if len(pooled_records) != len(matrix.runs) + len(matrix.references):
                raise ValueError("Pooled inputs lack mandatory configurations")
            observations = compare_metric_rows(json_value(expected), read_json(pooled)["rows"], context.replay_policy)
            values.append(result("pooled_metrics", observations))
        except (ValueError, KeyError, OSError) as exc:
            values.append(failure(str(pooled), exc))
    return emit_validation(merge_results("complete_matrix", values), context.output_root / "manifests/completion_validation.json")
