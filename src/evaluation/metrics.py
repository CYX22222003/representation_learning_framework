from __future__ import annotations

import torch


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    return {"mae": mae, "rmse": rmse}


def mse_and_corr(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_ = pred.reshape(-1)
    target_ = target.reshape(-1)
    mse = torch.mean((pred_ - target_) ** 2).item()
    corr = float("nan")
    if len(pred_) > 1:
        corr_tensor = torch.corrcoef(torch.stack([pred_, target_]))
        corr = float(corr_tensor[0, 1].item())
    return {"mse": mse, "corr": corr}


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits.reshape(-1))
    preds = (probs >= 0.5).float()
    y = labels.reshape(-1).float()

    tp = torch.sum((preds == 1) & (y == 1)).item()
    fp = torch.sum((preds == 1) & (y == 0)).item()
    fn = torch.sum((preds == 0) & (y == 1)).item()
    accuracy = torch.mean((preds == y).float()).item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"accuracy": accuracy, "f1": f1}


def multiclass_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> dict[str, object]:
    """Accuracy, macro-F1, per-class metrics, and confusion matrix for CE logits."""
    if n_classes <= 1:
        raise ValueError("n_classes must be greater than 1")
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError(f"Expected logits [N, {n_classes}], got {tuple(logits.shape)}")

    y = labels.reshape(-1).to(torch.long)
    if logits.shape[0] != y.shape[0]:
        raise ValueError(f"Logit/label row mismatch: {logits.shape[0]} vs {y.shape[0]}")

    preds = torch.argmax(logits, dim=1)
    accuracy = torch.mean((preds == y).float()).item()

    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for true, pred in zip(y.cpu(), preds.cpu()):
        if 0 <= int(true) < n_classes:
            cm[int(true), int(pred)] += 1

    per_class = []
    f1_values = []
    support_values = []
    for c in range(n_classes):
        tp = float(cm[c, c].item())
        fp = float(cm[:, c].sum().item() - cm[c, c].item())
        fn = float(cm[c, :].sum().item() - cm[c, c].item())
        support = int(cm[c, :].sum().item())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        per_class.append(
            {
                "class_index": c,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        f1_values.append(f1)
        support_values.append(support)

    total_support = max(sum(support_values), 1)
    weighted_f1 = sum(f1 * support for f1, support in zip(f1_values, support_values)) / total_support
    return {
        "accuracy": accuracy,
        "macro_f1": float(sum(f1_values) / n_classes),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
