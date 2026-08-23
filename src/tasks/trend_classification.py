from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import (  # noqa: F401  (re-exported for callers)
    classification_metrics,
    multiclass_classification_metrics,
)


def build_trend_labels(
    sequences: np.ndarray, price_index: int = 3, horizon: int = 1, threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) pairs for binary trend classification.

    X[i] = sequences[i]   — current window (input)
    y[i] = 1 if the close price at the end of the next window is higher
           than the close price at the end of the current window by more
           than `threshold` (as a fraction), else 0.

    threshold=0.0 means any upward move is labelled 1.
    """
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, seq_len, features], got {arr.shape}")
    if len(arr) <= horizon:
        raise ValueError("Not enough sequences for requested horizon")

    # Last timestep close of the current window vs. last timestep close of the next window.
    current = arr[:-horizon, -1, price_index]
    future = arr[horizon:, -1, price_index]
    ret = (future - current) / np.clip(np.abs(current), 1e-8, None)
    labels = (ret > threshold).astype(np.float32)
    return arr[:-horizon], labels


class TrendClassifier(nn.Module):
    """
    Default decoder for trend classification (task head).

    By default this preserves the original binary path: one logit trained with
    BCEWithLogitsLoss. For the TA-MLP-aligned MVP trend task, pass
    n_classes=3 and train with CrossEntropyLoss.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 1) -> None:
        super().__init__()
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
