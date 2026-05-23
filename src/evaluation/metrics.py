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
