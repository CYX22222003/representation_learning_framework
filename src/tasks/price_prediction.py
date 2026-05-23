from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from evaluation.metrics import regression_metrics  # noqa: F401  (re-exported for callers)


def build_price_prediction_targets(
    sequences: np.ndarray, price_index: int = 3, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) where X is current sequence and y is future close price.
    """
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, seq_len, features], got {arr.shape}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(arr) <= horizon:
        raise ValueError("Not enough sequences for requested horizon")

    X = arr[:-horizon]
    y = arr[horizon:, -1, price_index]
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class PriceRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
