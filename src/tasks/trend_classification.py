from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_trend_labels(
    sequences: np.ndarray, price_index: int = 3, horizon: int = 1, threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary label: future end price movement > threshold.
    """
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, seq_len, features], got {arr.shape}")
    if len(arr) <= horizon:
        raise ValueError("Not enough sequences for requested horizon")

    current = arr[:-horizon, -1, price_index]
    future = arr[horizon:, -1, price_index]
    ret = (future - current) / np.clip(np.abs(current), 1e-8, None)
    labels = (ret > threshold).astype(np.float32)
    return arr[:-horizon], labels


class TrendClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
