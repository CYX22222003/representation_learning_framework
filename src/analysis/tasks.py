"""Existing task heads/losses and raw-unit metric adapters."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.stats import rankdata

from .types import AnalysisTask, Array, MetricDirection, MetricResult, PredictionRecord, ProbeRunSpec, TaskDataset

PRIMARY_METRICS = {
    AnalysisTask.CLASSIFICATION: ("macro_f1", "balanced_accuracy"),
    AnalysisTask.PRICE: ("mae", "rmse", "rank_ic"),
    AnalysisTask.VARIANCE: ("mae", "rmse", "spearman"),
}


def build_head(spec: ProbeRunSpec):
    if spec.source.task == AnalysisTask.CLASSIFICATION:
        from tasks.trend_classification import TrendClassifier
        return TrendClassifier(spec.ablation.width, hidden_dim=128, n_classes=3)
    if spec.source.task == AnalysisTask.PRICE:
        from training.phase5_absolute_price import AbsolutePriceRegressor
        return AbsolutePriceRegressor(spec.ablation.width, hidden_dim=128)
    from tasks.volatility_prediction import VolatilityRegressor
    return VolatilityRegressor(spec.ablation.width, hidden_dim=128)


def build_loss(spec: ProbeRunSpec, training_targets: Array):
    import torch.nn as nn
    if spec.source.task == AnalysisTask.CLASSIFICATION:
        from tasks.phase2_classification.protocols import LogitAdjustedCrossEntropy, class_priors
        return LogitAdjustedCrossEntropy(class_priors(training_targets, 3), 1.0)
    return nn.SmoothL1Loss(beta=1.0) if spec.source.task == AnalysisTask.VARIANCE else nn.MSELoss()


def metric(name, value, count, reason=None):
    lower = name in ("mae", "mse", "rmse", "nll", "multiclass_brier")
    available = value is not None and np.isfinite(value)
    return MetricResult(name, float(value) if available else None,
                        MetricDirection.LOWER if lower else MetricDirection.HIGHER, count,
                        "available" if available else "unavailable",
                        None if available else reason or "undefined_for_constant_or_insufficient_data")


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def validate_record(record: PredictionRecord) -> None:
    from .sources import validate_identity
    validate_identity(record.identity)
    n = len(record.targets)
    if record.task != record.identity.task or record.walk != record.identity.walk:
        raise ValueError("Prediction task/walk identity mismatch")
    if record.targets.shape != (n,) or len(record.outputs) != n:
        raise ValueError("Prediction/target shape mismatch")
    if not np.isfinite(record.targets).all() or not np.isfinite(record.outputs).all():
        raise ValueError("Non-finite prediction or target")
    if any(len(v) != n for v in record.metadata.values()):
        raise ValueError("Prediction metadata rows mismatch")
    if record.task == AnalysisTask.CLASSIFICATION:
        if record.units != "class_logits" or record.outputs.shape != (n, 3):
            raise ValueError("Classification requires raw three-class logits")
        if np.any((record.targets < 0) | (record.targets > 2) | (record.targets != record.targets.astype(int))):
            raise ValueError("Invalid classification targets")
    else:
        if record.outputs.shape not in ((n,), (n, 1)):
            raise ValueError("Regression requires one output per row")
        values = record.outputs.reshape(-1)
        if record.task == AnalysisTask.PRICE:
            if record.units not in ("raw_probability", "movement_score"):
                raise ValueError("Price output unit mismatch")
            if record.units == "raw_probability" and np.any((values < 0) | (values > 1)):
                raise ValueError("Price predictions outside probability range")
        elif record.units != "raw_realised_variance" or np.any(values < 0) or np.any(record.targets < 0):
            raise ValueError("RV outputs/targets must be nonnegative raw variance")


def rank_ic_values(record: PredictionRecord) -> dict[int, float]:
    if record.task != AnalysisTask.PRICE:
        return {}
    current = record.metadata["current_close"]
    scores = record.outputs.reshape(-1) if record.units == "movement_score" else record.outputs.reshape(-1) - current
    target = record.targets - current
    times, conditions = record.identity.decision_date_ns, record.identity.condition_ids
    if len(set(zip(conditions.tolist(), times.tolist()))) != len(times):
        raise ValueError("Duplicate condition/timestamp in Rank IC")
    result = {}
    for timestamp in np.unique(times):
        mask = times == timestamp
        if len(np.unique(conditions[mask])) < 5:
            continue
        value = correlation(rankdata(scores[mask]), rankdata(target[mask]))
        if value is not None and np.isfinite(value):
            result[int(timestamp)] = value
    return result


def evaluate_predictions(record: PredictionRecord) -> tuple[MetricResult, ...]:
    validate_record(record)
    y, pred = record.targets, record.outputs
    n = len(y)
    if not n:
        return tuple(metric(name, None, 0, "empty_group") for name in PRIMARY_METRICS[record.task])
    if record.task == AnalysisTask.CLASSIFICATION:
        from tasks.phase2_classification.metrics import probabilistic_classification_metrics
        values, _ = probabilistic_classification_metrics(pred, y, ("DOWN", "STABLE", "UP"))
        values["weighted_f1"] = sum(c["f1"] * c["support"] for c in values["per_class"]) / n
        result = [metric(k, v, n) for k, v in values.items() if np.isscalar(v)]
        for cls in values["per_class"]:
            for name in ("precision", "recall", "f1", "roc_auc", "average_precision"):
                result.append(metric(f"{cls['class_name'].lower()}_{name}", cls[name], n))
        recalls = [c["recall"] for c in values["per_class"] if c["support"] > 0]
        result.append(metric("observed_class_balanced_accuracy", np.mean(recalls), n))
        return tuple(result)
    pred = pred.reshape(-1).astype(np.float64)
    result = []
    if record.units != "movement_score":
        err = pred - y
        result = [metric("mae", np.abs(err).mean(), n), metric("mse", np.square(err).mean(), n),
                  metric("rmse", np.sqrt(np.square(err).mean()), n), metric("bias", err.mean(), n),
                  metric("pearson", correlation(pred, y), n),
                  metric("spearman", correlation(rankdata(pred), rankdata(y)), n)]
    if record.task == AnalysisTask.PRICE:
        current = record.metadata["current_close"]
        scores = pred if record.units == "movement_score" else pred - current
        actual = y - current
        result += [metric("implied_pearson", correlation(scores, actual), n),
                   metric("implied_spearman", correlation(rankdata(scores), rankdata(actual)), n),
                   metric("sign_agreement", np.mean(np.sign(scores) == np.sign(actual)), n)]
        nonzero = actual != 0
        result.append(metric("nonzero_sign_agreement", np.mean(np.sign(scores[nonzero]) == np.sign(actual[nonzero])) if nonzero.any() else None, int(nonzero.sum())))
        ic = np.asarray(list(rank_ic_values(record).values()))
        result += [metric("rank_ic", ic.mean() if len(ic) else None, len(ic)),
                   metric("rank_ic_median", np.median(ic) if len(ic) else None, len(ic)),
                   metric("rank_ic_positive_fraction", np.mean(ic > 0) if len(ic) else None, len(ic))]
    return tuple(result)


def reference_predictions(dataset: TaskDataset) -> dict[str, Array]:
    n = len(dataset.test.targets)
    if dataset.source.task == AnalysisTask.CLASSIFICATION:
        from tasks.phase2_classification.protocols import class_priors
        stable = np.full((n, 3), -30., dtype=np.float32)
        stable[:, 1] = 0.
        return {"always_stable": stable, "training_priors": np.broadcast_to(np.log(class_priors(dataset.train.targets)), (n, 3)).copy()}
    if dataset.source.task == AnalysisTask.PRICE:
        return {"persistence": dataset.test.metadata["current_close"],
                "last_hour_reversal": dataset.test.metadata["last_hour_reversal"]}
    return {"zero": np.zeros(n), "training_median": np.full(n, np.median(dataset.train.targets)),
            "historical_rv": dataset.test.metadata["historical_rv"]}


def reference_records(dataset: TaskDataset, epoch=50) -> tuple[PredictionRecord, ...]:
    units = {AnalysisTask.CLASSIFICATION: "class_logits", AnalysisTask.PRICE: "raw_probability",
             AnalysisTask.VARIANCE: "raw_realised_variance"}[dataset.source.task]
    return tuple(PredictionRecord(dataset.source.task, dataset.source.walk, name, 0, epoch,
        dataset.test.identity, dataset.test.targets, values,
        "movement_score" if name == "last_hour_reversal" else units,
        dataset.source_hashes, dataset.test.metadata) for name, values in reference_predictions(dataset).items())
