"""Ordered experiment stages with isolated failures and explicit execution."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import (freeze_json, hash_array, identity_arrays, identity_hash, json_value,
                        load_prediction, read_json, write_arrays_atomic, write_json_atomic)
from .contracts import BRANCHES, build_matrix
from .features import fit_standardizer, load_standardizer, save_standardizer, select_features, validate_scaler_parity
from .types import AnalysisContext, AnalysisTask, ExperimentMatrix, ReportResult, ValidationResult
from .validation import emit_validation, failure, merge_results, result


class AnalysisPipeline:
    """Continue independent entries; invalid shared inputs block their dependents."""

    def __init__(self, context: AnalysisContext):
        self.context = context
        self._cached = None

    def _dataset(self, source):
        from .sources import load_dataset
        if self._cached is None or self._cached.source != source:
            self._cached = load_dataset(source, self.context)
        return self._cached

    def _event(self, stage, state, **details):
        import datetime
        import sys
        line = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "stage": stage, "state": state, **details}
        print(json.dumps(line, sort_keys=True), file=sys.stderr, flush=True)
        path = self.context.output_root / "logs/events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")

    def _freeze_policy(self):
        freeze_json(self.context.output_root / "manifests/replay_policy.json", json_value(self.context.replay_policy))

    def _admit(self):
        from .runtime import collect_environment, preflight
        from .training import source_code_hashes
        self._freeze_policy()
        environment = collect_environment(self.context)
        validation = emit_validation(preflight(self.context, environment))
        if not validation.usable:
            raise ValueError(f"Lumid preflight failed: {dict(validation.checks)}")
        freeze_json(self.context.output_root / "manifests/preflight.json",
                    {"environment": environment, "validation": validation,
                     "code_hashes": source_code_hashes(self.context)})
        return environment

    def audit(self) -> ValidationResult:
        from .runtime import inventory_dependencies
        from .sources import audit_source
        from .replay import replay_h0
        self._freeze_policy()
        matrix = build_matrix(self.context)
        validations, inventory = [], {}
        for source in matrix.references:
            self._event("audit", "start", task=source.task.value, walk=source.walk)
            try:
                dependencies = inventory_dependencies((source,), self.context)
                inventory.update({str(p.relative_path): p for p in dependencies})
                source_result = audit_source(source, self.context)
                validations.append(source_result)
                if source_result.usable:
                    validations.append(replay_h0(source, self.context, self._dataset(source)))
                else:
                    emit_validation(source_result)
            except Exception as exc:
                validations.append(emit_validation(failure(str(source.dataset_path), exc)))
        write_json_atomic(self.context.output_root / "manifests/dependency_inventory.json",
                          {"files": list(inventory.values())})
        value = emit_validation(merge_results("source_audit", validations),
                                self.context.output_root / "manifests/source_audit.json")
        self._event("audit", value.status)
        return value

    def prepare(self) -> ExperimentMatrix:
        from .conditional import fit_conditions
        self._freeze_policy()
        matrix = build_matrix(self.context)
        entries, validations = [], []
        for source in matrix.references:
            try:
                data = self._dataset(source)
                canonical = load_standardizer(source.h0_scaler_path)
                base = self.context.output_root / "features" / source.task.value / f"walk{source.walk}"
                freeze_json(base / "conditions.json", json_value(fit_conditions(data)))
                for spec in (s for s in matrix.runs if s.source == source):
                    scaler = fit_standardizer(select_features(data.train, spec.ablation), data.train.identity)
                    validate_scaler_parity(scaler, canonical, spec.ablation)
                    destination = base / spec.ablation.configuration
                    freeze_json(destination / "branch_view.json", {
                        "branches": spec.ablation.branches, "width": spec.ablation.width,
                        "source_hashes": data.source_hashes,
                        "train_identity": identity_hash(data.train.identity), "test_identity": identity_hash(data.test.identity),
                        "training_targets": hash_array(data.train.targets), "evaluation_targets": hash_array(data.test.targets),
                        "scaler_arrays": {k: hash_array(getattr(scaler, k)) for k in ("mean", "raw_std", "scale")}})
                    if not (destination / "feature_standardizer.npz").exists():
                        save_standardizer(destination / "feature_standardizer.npz", scaler)
                    entries.append(json_value(spec))
                validations.append(result(str(source.feature_path)))
            except Exception as exc:
                validations.append(emit_validation(failure(str(source.feature_path), exc)))
        freeze_json(self.context.output_root / "manifests/matrix_seed0.json", {
            "runs": json_value(matrix.runs), "references": json_value(matrix.references),
            "expected_new_runs": 60, "expected_reference_runs": 6, "expected_snapshots": 198})
        emit_validation(merge_results("prepared_sources", validations),
                        self.context.output_root / "manifests/preparation_status.json")
        return matrix

    def launch(self, *, device: str = "cuda", execute: bool = False) -> None:
        from .training import ProbeRunner
        from .replay import replay_h0
        environment = self._admit() if execute else None
        matrix = self.prepare()
        if not execute:
            self._event("launch", "manifest_only", new_runs=len(matrix.runs), training_started=False)
            return
        if device != "cuda":
            raise ValueError("Production matrix is admitted for Lumid CUDA; CPU is for development checks")
        runner = ProbeRunner(self.context, environment)
        for source in matrix.references:
            try:
                data = self._dataset(source)
                from .sources import audit_source
                audited = emit_validation(audit_source(source, self.context))
                if not audited.usable:
                    raise ValueError("Shared source lineage failed audit")
                if not replay_h0(source, self.context, data).usable:
                    raise ValueError("H0 reference is invalid")
            except Exception as exc:
                self._event("source", "blocked", task=source.task.value, walk=source.walk, reason=str(exc))
                continue
            for original in (s for s in matrix.runs if s.source == source):
                spec = replace(original, device=device)
                self._event("train", "start", task=source.task.value, configuration=spec.ablation.configuration, walk=source.walk)
                try:
                    value = runner.train(spec, data)
                    self._event("train", value.state.value, configuration=spec.ablation.configuration, walk=source.walk)
                except Exception as exc:
                    self._event("train", "failed", configuration=spec.ablation.configuration,
                                walk=source.walk, reason=str(exc), task=source.task.value)

    def similarity(self) -> None:
        self._freeze_policy()
        from .sources import locate_source, load_dataset
        from .similarity import compute_cka, sample_classification_splits
        from .replay import replay_similarity
        data = [load_dataset(locate_source(self.context, AnalysisTask.CLASSIFICATION, w), self.context) for w in (1, 2)]
        samples = sample_classification_splits(data)
        for dataset in data:
            scaler = fit_standardizer(np.concatenate([dataset.train.branches[b.key] for b in BRANCHES], axis=1), dataset.train.identity)
            for sample in (s for s in samples if s.walk == dataset.source.walk):
                split = dataset.train if sample.split.value == "train" else dataset.test
                value = compute_cka(split, scaler, sample)
                base = self.context.output_root / f"analysis/cka/classification_h2/walk{sample.walk}/{sample.split.value}"
                write_arrays_atomic(base / "scores.npz", {"scores": value.scores})
                write_arrays_atomic(base / "sample.npz", {"positions": sample.positions, **identity_arrays(sample.identity)})
                finite = np.isfinite(value.scores)
                write_json_atomic(base / "result.json", {
                    "branches": value.branches, "unavailable_reasons": value.unavailable_reasons,
                    "scores": [[float(v) if np.isfinite(v) else None for v in row] for row in value.scores],
                    "sample_identity_hash": sample.identity_hash, "sample_size": len(sample.positions),
                    "positions_sha256": hash_array(sample.positions), "scores_sha256": hash_array(value.scores),
                    "source_hashes": dataset.source_hashes})
                emit_validation(replay_similarity(value, dataset), base / "replay_validation.json")

    def conditional(self) -> None:
        self._freeze_policy()
        from .conditional import assign_groups, evaluate_groups, fit_conditions, summarize_contracts, pool_walks
        from .sources import load_h0_predictions
        from .replay import replay_h0, replay_snapshot
        matrix = build_matrix(self.context)
        pooled_records = []
        for source in matrix.references:
            rows, configs = [], []
            try:
                dataset = self._dataset(source)
                conditions = fit_conditions(dataset)
                records = []
                if replay_h0(source, self.context, dataset).usable:
                    records.append(load_h0_predictions(source, 50, self.context, dataset))
                for spec in (s for s in matrix.runs if s.source == source):
                    if replay_snapshot(spec, dataset, 50, context=self.context).usable:
                        records.append(load_prediction(spec.output_path / "e50/predictions.npz"))
                for record in records:
                    grouped = evaluate_groups(record, assign_groups(record, conditions))
                    rows += list(grouped) + list(summarize_contracts(grouped))
                    configs.append(record.configuration)
                    sparse = sum(bool(row.diagnostics.get("sparse")) for row in grouped)
                    if sparse:
                        self._event("conditional", "warning", configuration=record.configuration,
                                    sparse_groups=sparse, task=source.task.value, walk=source.walk)
                pooled_records.extend(records)
                write_json_atomic(self.context.output_root / "analysis/conditional" / source.task.value / f"walk{source.walk}/results.json",
                                  {"conditions": conditions, "source_hashes": dataset.source_hashes,
                                   "configurations": sorted(configs), "rows": rows})
            except Exception as exc:
                self._event("conditional", "failed", task=source.task.value, walk=source.walk, reason=str(exc))
        by_config = {}
        for record in pooled_records:
            by_config.setdefault((record.task, record.configuration), []).append(record)
        pooled = []
        for pair in by_config.values():
            if len(pair) == 2:
                pooled.extend(pool_walks(pair))
        write_json_atomic(self.context.output_root / "analysis/conditional/pooled.json", {"rows": pooled})

    def validate(self, stage: str = "all") -> ValidationResult:
        self._freeze_policy()
        from .replay import validate_complete, validate_preparation, replay_snapshot
        matrix = build_matrix(self.context)
        if stage == "preparation":
            return validate_preparation(self.context, matrix)
        if stage == "all":
            return validate_complete(self.context, matrix)
        if stage != "runs":
            raise ValueError("Unknown validation stage")
        checks = []
        for source in matrix.references:
            try:
                data = self._dataset(source)
                for spec in (s for s in matrix.runs if s.source == source):
                    for epoch in spec.recipe.snapshots:
                        checks.append(replay_snapshot(spec, data, epoch, context=self.context))
            except Exception as exc:
                checks.append(failure(str(source.dataset_path), exc))
        return emit_validation(merge_results("runs", checks), self.context.output_root / "manifests/run_validation.json")

    def report(self, *, allow_partial: bool = False) -> ReportResult:
        self._freeze_policy()
        from .reporting import build_report
        return build_report(self.context, build_matrix(self.context), allow_partial=allow_partial)

    def run(self, *, device: str = "cuda", execute: bool = False) -> ReportResult:
        if not execute:
            self.launch(device=device, execute=False)
            return ReportResult(self.context.output_root, False, ())
        self._admit()
        self.audit()
        self.prepare()
        try:
            self.similarity()
        except Exception as exc:
            self._event("similarity", "failed", reason=str(exc))
        self.launch(device=device, execute=True)
        self.validate("runs")
        self.conditional()
        self.validate("all")
        return self.report(allow_partial=True)
