"""Standalone skill and paired, benefit-positive H0 removal effects."""

from dataclasses import replace
from typing import Sequence

import numpy as np

from .artifacts import identity_hash
from .tasks import evaluate_predictions, metric, rank_ic_values, validate_record
from .types import AnalysisTask, ComparisonResult, MetricDirection, MetricRow, PredictionRecord


def assert_paired_predictions(left: PredictionRecord, right: PredictionRecord) -> None:
    validate_record(left)
    validate_record(right)
    if (left.task, left.walk, left.epoch) != (right.task, right.walk, right.epoch):
        raise ValueError("Paired prediction task/walk/epoch mismatch")
    if identity_hash(left.identity) != identity_hash(right.identity) or not np.array_equal(left.targets, right.targets):
        raise ValueError("Paired predictions have different ordered identities or targets")


def summarize_predictions(records: Sequence[PredictionRecord]) -> tuple[MetricRow, ...]:
    rows = []
    for r in records:
        diagnostics = {"rows": len(r.targets), "contracts": len(np.unique(r.identity.condition_ids))}
        if r.task == AnalysisTask.CLASSIFICATION:
            predicted = r.outputs.argmax(axis=1)
            matrix = np.zeros((3, 3), dtype=np.int64)
            np.add.at(matrix, (r.targets.astype(int), predicted), 1)
            diagnostics.update(confusion_matrix=matrix.tolist(),
                               observed_class_counts=matrix.sum(axis=1).tolist(),
                               predicted_class_counts=matrix.sum(axis=0).tolist())
        rows.append(MetricRow(r.task, r.walk, r.configuration, r.epoch, "overall", evaluate_predictions(r), diagnostics))
    return tuple(rows)


def compare(candidate: PredictionRecord, reference: PredictionRecord) -> tuple[ComparisonResult, ...]:
    assert_paired_predictions(candidate, reference)
    c, r = summarize_predictions((candidate, reference))
    # Nonlinear IC comparisons must share valid cross-sectional timestamps.
    if candidate.task == AnalysisTask.PRICE:
        cv, rv = rank_ic_values(candidate), rank_ic_values(reference)
        common = sorted(cv.keys() & rv.keys())
        c = replace(c, metrics=tuple(metric("rank_ic", np.mean([cv[t] for t in common]) if common else None, len(common)) if m.name == "rank_ic" else m for m in c.metrics))
        r = replace(r, metrics=tuple(metric("rank_ic", np.mean([rv[t] for t in common]) if common else None, len(common)) if m.name == "rank_ic" else m for m in r.metrics))
    refs = {m.name: m for m in r.metrics}
    result = []
    for m in c.metrics:
        if m.name == "bias":
            continue  # Signed bias has no general higher/lower-is-better ordering.
        if m.name not in refs:
            continue
        baseline = refs[m.name]
        available = m.value is not None and baseline.value is not None
        delta = ((m.value - baseline.value) if m.direction == MetricDirection.HIGHER else
                 (baseline.value - m.value)) if available else None
        relative = delta / baseline.value if available and m.direction == MetricDirection.LOWER and baseline.value > 0 else None
        result.append(ComparisonResult(c, r, m.name, delta, relative,
                                       None if available else "undefined_paired_metric"))
    return tuple(result)


def single_branch_skill(single: PredictionRecord, reference: PredictionRecord) -> tuple[ComparisonResult, ...]:
    return compare(single, reference)


def removal_contribution(h0: PredictionRecord, removal: PredictionRecord) -> tuple[ComparisonResult, ...]:
    values = compare(h0, removal)
    output = []
    for value in values:
        metric = next(m for m in value.candidate.metrics if m.name == value.metric)
        denominator = metric.value
        relative = (value.benefit_positive_difference / denominator
                    if metric.direction == MetricDirection.LOWER and denominator is not None
                    and denominator > 0 and value.benefit_positive_difference is not None else None)
        output.append(replace(value, relative_error_change=relative))
    return tuple(output)
