from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import classification_metrics  # noqa: F401  (re-exported for callers)


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
