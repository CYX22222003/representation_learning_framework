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
    Create (X, y) pairs for price prediction.

    X[i] = sequences[i]           — current window (input features)
    y[i] = sequences[i+horizon, -1, price_index]  — close price at end of next window

    price_index=3 selects the 'close' column (OHLCV order: 0=open,1=high,2=low,3=close,4=volume).
    horizon=1 means predict one window ahead.
    """
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, seq_len, features], got {arr.shape}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(arr) <= horizon:
        raise ValueError("Not enough sequences for requested horizon")

    X = arr[:-horizon]
    # Take the last timestep of the future window as the price target.
    y = arr[horizon:, -1, price_index]
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that pairs pre-extracted feature vectors with scalar targets."""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class PriceRegressor(nn.Module):
    """
    Default decoder for price prediction (task head).

    Lightweight 3-layer MLP that maps aggregator embeddings → scalar price output.
    Intentionally simple: representation quality, not decoder depth, drives performance.
    """

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