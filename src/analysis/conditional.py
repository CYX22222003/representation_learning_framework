"""Separate walk/contract/future-RV groups, preserving full evaluation rows."""

from dataclasses import replace
from typing import Sequence

import numpy as np

from .artifacts import hash_array, subset_record
from .tasks import evaluate_predictions, metric
from .types import (AnalysisTask, ConditionGroup, ConditionSpec, MetricRow, PredictionRecord,
                    RowIdentity, TaskDataset)


def fit_conditions(dataset: TaskDataset) -> ConditionSpec:
    if dataset.source.task != AnalysisTask.VARIANCE:
        return ConditionSpec(dataset.source.task, dataset.source.walk)
    q90, q99 = np.quantile(dataset.train.targets, [0.9, 0.99], method="linear")
    return ConditionSpec(dataset.source.task, dataset.source.walk,
        ("calendar", "contract", "future_realised_variance"), float(q90), float(q99),
        hash_array(dataset.train.targets))


def assign_groups(record: PredictionRecord, conditions: ConditionSpec) -> tuple[ConditionGroup, ...]:
    if (record.task, record.walk) != (conditions.task, conditions.walk):
        raise ValueError("Condition task/walk mismatch")
    allowed = {"calendar", "contract", "future_realised_variance"}
    if set(conditions.dimensions) - allowed:
        raise ValueError("Uncommissioned condition grouping")
    groups = []
    if "calendar" in conditions.dimensions:
        groups.append(ConditionGroup("calendar", f"walk{record.walk}", np.ones(len(record.targets), bool)))
    if "contract" in conditions.dimensions:
        groups += [ConditionGroup("contract", str(k), record.identity.condition_ids == k)
                   for k in np.unique(record.identity.condition_ids)]
    if "future_realised_variance" in conditions.dimensions:
        if record.task != AnalysisTask.VARIANCE or conditions.rv_q90 is None or conditions.rv_q99 is None or not conditions.training_target_hash:
            raise ValueError("RV groups require training-derived thresholds")
        y, q90, q99 = record.targets, conditions.rv_q90, conditions.rv_q99
        groups += [ConditionGroup("rv", "below_q90", y < q90),
                   ConditionGroup("rv", "q90_to_q99", (y >= q90) & (y < q99)),
                   ConditionGroup("rv", "at_or_above_q99", y >= q99),
                   ConditionGroup("rv_tail", "q90", y >= q90),
                   ConditionGroup("rv_tail", "q99", y >= q99)]
    return tuple(groups)


def evaluate_groups(record: PredictionRecord, groups: Sequence[ConditionGroup]) -> tuple[MetricRow, ...]:
    rows = []
    total_sse = float(np.square(record.outputs.reshape(-1) - record.targets).sum()) if record.task == AnalysisTask.VARIANCE else None
    for group in groups:
        if group.mask.dtype != bool or group.mask.shape != record.targets.shape:
            raise ValueError("Condition mask mismatch")
        selected = subset_record(record, group.mask)
        metrics = evaluate_predictions(selected)
        n = len(selected.targets)
        contracts = len(np.unique(selected.identity.condition_ids))
        diagnostics = {"rows": n, "contracts": contracts, "mask_sha256": hash_array(group.mask),
                       "sparse": n < 100 or contracts < 3, "case_specific": group.dimension == "contract"}
        if record.task == AnalysisTask.CLASSIFICATION:
            counts = np.bincount(selected.targets.astype(int), minlength=3)
            diagnostics.update(class_counts=counts.tolist(), missing_classes=np.flatnonzero(counts == 0).tolist())
        if group.dimension == "rv_tail":
            sse = float(np.square(selected.outputs.reshape(-1) - selected.targets).sum())
            metrics += (metric("tail_row_fraction", n / len(record.targets), n),
                        metric("tail_sse_share", sse / total_sse if total_sse else None, n, "zero_total_sse"))
        rows.append(MetricRow(record.task, record.walk, record.configuration, record.epoch,
                              f"{group.dimension}:{group.name}", metrics, diagnostics))
    return tuple(rows)


def summarize_contracts(rows: Sequence[MetricRow]) -> tuple[MetricRow, ...]:
    grouped = {}
    for row in rows:
        if row.group.startswith("contract:"):
            grouped.setdefault((row.task, row.walk, row.configuration, row.epoch), []).append(row)
    output = []
    for (task, walk, config, epoch), contracts in grouped.items():
        subsets = [("contract_macro", contracts)]
        if task == AnalysisTask.CLASSIFICATION:
            subsets.append(("contract_macro_all_classes", [r for r in contracts if not r.diagnostics.get("missing_classes")]))
        for label, subset in subsets:
            names = sorted({m.name for r in contracts for m in r.metrics})
            values = []
            excluded = {}
            for name in names:
                valid = [m.value for r in subset for m in r.metrics if m.name == name and m.value is not None]
                values.append(metric(name, np.mean(valid) if valid else None, len(valid), "no_defined_contract_metrics"))
                excluded[name] = len(subset) - len(valid)
            output.append(MetricRow(task, walk, config, epoch, label, tuple(values),
                                    {"contracts": len(subset), "excluded_undefined": excluded}))
    return tuple(output)


def pool_walks(records: Sequence[PredictionRecord]) -> tuple[MetricRow, ...]:
    grouped = {}
    for record in records:
        grouped.setdefault((record.task, record.configuration, record.epoch), []).append(record)
    rows = []
    for (task, config, epoch), pair in grouped.items():
        if len(pair) != 2 or {r.walk for r in pair} != {1, 2}:
            raise ValueError("Pooling requires exactly the two walks")
        pair.sort(key=lambda r: r.walk)
        if max(pair[0].identity.decision_date_ns) >= min(pair[1].identity.decision_date_ns):
            raise ValueError("Walk evaluation periods overlap")
        fields = set(pair[0].identity.target_fields)
        if fields != set(pair[1].identity.target_fields):
            raise ValueError("Pooled target identity schema mismatch")
        identity = RowIdentity(task, 0,
            np.concatenate([np.asarray([f"walk{r.walk}:{k}" for k in r.identity.condition_ids]) for r in pair]),
            *(np.concatenate([getattr(r.identity, f) for r in pair]) for f in ("window_start_ns", "decision_date_ns", "decision_availability_ns")),
            {k: np.concatenate([r.identity.target_fields[k] for r in pair]) for k in fields})
        metadata = {k: np.concatenate([r.metadata[k] for r in pair]) for k in pair[0].metadata.keys() & pair[1].metadata.keys()}
        pooled = replace(pair[0], walk=0, identity=identity, targets=np.concatenate([r.targets for r in pair]),
                         outputs=np.concatenate([r.outputs for r in pair]), metadata=metadata)
        rows.append(MetricRow(task, None, config, epoch, "pooled", evaluate_predictions(pooled)))
        individual = [evaluate_predictions(r) for r in pair]
        means = tuple(metric(m.name, np.mean([next(x.value for x in v if x.name == m.name) for v in individual])
                             if all(next(x.value for x in v if x.name == m.name) is not None for v in individual) else None, 2)
                      for m in individual[0])
        rows.append(MetricRow(task, None, config, epoch, "equal_walk_mean", means))
        contract_rows = evaluate_groups(pooled, tuple(ConditionGroup("contract", str(k), identity.condition_ids == k) for k in np.unique(identity.condition_ids)))
        rows += list(summarize_contracts(contract_rows))
    return tuple(rows)
