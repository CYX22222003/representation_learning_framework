from __future__ import annotations

import numpy as np


def _binary_roc_curve(target: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    y = target[order].astype(np.int64)
    s = score[order]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan])
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = 1 + distinct - tps
    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    thresholds = np.r_[np.inf, s[distinct], -np.inf]
    return fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)


def _binary_pr_curve(target: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    y = target[order].astype(np.int64)
    s = score[order]
    positives = int(y.sum())
    if positives == 0:
        return np.asarray([np.nan]), np.asarray([np.nan]), np.asarray([np.nan])
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = 1 + distinct - tps
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    return precision.astype(np.float64), recall.astype(np.float64), s[distinct].astype(np.float64)


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = _binary_pr_curve(target, score)
    if np.isnan(precision).any():
        return float("nan")
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def probabilistic_classification_metrics(
    logits: np.ndarray, labels: np.ndarray, class_names: list[str] | tuple[str, ...]
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    raw = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_classes = len(class_names)
    if raw.ndim != 2 or raw.shape != (len(y), n_classes):
        raise ValueError(f"Expected logits {(len(y), n_classes)}, got {raw.shape}")
    if len(y) == 0 or np.any((y < 0) | (y >= n_classes)):
        raise ValueError("labels are empty or outside the class range")
    shifted = raw - raw.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, predictions), 1)

    per_class: list[dict[str, object]] = []
    aucs: list[float] = []
    aps: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []
    curves: dict[str, np.ndarray] = {}
    for class_id, class_name in enumerate(class_names):
        tp = int(cm[class_id, class_id])
        fp = int(cm[:, class_id].sum() - tp)
        fn = int(cm[class_id, :].sum() - tp)
        support = int(cm[class_id, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        binary = (y == class_id).astype(np.int64)
        fpr, tpr, roc_thresholds = _binary_roc_curve(binary, probabilities[:, class_id])
        roc_auc = float(np.trapezoid(tpr, fpr)) if not np.isnan(fpr).any() else float("nan")
        pr_precision, pr_recall, pr_thresholds = _binary_pr_curve(binary, probabilities[:, class_id])
        average_precision = _average_precision(binary, probabilities[:, class_id])
        key = str(class_name).lower()
        curves[f"{key}_roc_fpr"] = fpr.astype(np.float32)
        curves[f"{key}_roc_tpr"] = tpr.astype(np.float32)
        curves[f"{key}_roc_thresholds"] = roc_thresholds.astype(np.float32)
        curves[f"{key}_pr_precision"] = pr_precision.astype(np.float32)
        curves[f"{key}_pr_recall"] = pr_recall.astype(np.float32)
        curves[f"{key}_pr_thresholds"] = pr_thresholds.astype(np.float32)
        per_class.append(
            {
                "class_index": class_id, "class_name": str(class_name), "precision": precision,
                "recall": recall, "f1": f1, "support": support, "roc_auc": roc_auc,
                "average_precision": average_precision,
            }
        )
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        aucs.append(roc_auc)
        aps.append(average_precision)

    one_hot = np.eye(n_classes, dtype=np.float64)[y]
    nll = float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    predicted_counts = np.bincount(predictions, minlength=n_classes)
    observed_counts = np.bincount(y, minlength=n_classes)
    finite_aucs = [value for value in aucs if np.isfinite(value)]
    finite_aps = [value for value in aps if np.isfinite(value)]
    metrics: dict[str, object] = {
        "accuracy": float((predictions == y).mean()),
        "macro_f1": float(np.mean(f1s)),
        "weighted_f1": float(np.average(f1s, weights=np.maximum(supports, 1))),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_roc_auc": float(np.mean(finite_aucs)) if finite_aucs else float("nan"),
        "macro_average_precision": float(np.mean(finite_aps)) if finite_aps else float("nan"),
        "nll": nll,
        "multiclass_brier": brier,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "observed_class_counts": observed_counts.tolist(),
        "predicted_class_counts": predicted_counts.tolist(),
        "observed_class_proportions": (observed_counts / len(y)).tolist(),
        "predicted_class_proportions": (predicted_counts / len(y)).tolist(),
    }
    arrays = {
        "logits": raw.astype(np.float32),
        "probabilities": probabilities.astype(np.float32),
        "predictions": predictions.astype(np.int64),
        "targets": y.astype(np.int64),
        "confusion_matrix": cm,
        **curves,
    }
    return metrics, arrays
