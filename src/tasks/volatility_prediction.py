from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics import mse_and_corr  # noqa: F401  (re-exported for callers)


def realized_volatility(prices: np.ndarray, eps: float = 1e-8) -> float:
    """
    Compute realised volatility as the RMS of log-returns within a price window.

    This is the standard definition used in GINN and related volatility papers:
    σ_realised = sqrt( mean( r_t² ) ) where r_t = log(P_t / P_{t-1}).
    """
    p = np.asarray(prices, dtype=np.float32)
    if len(p) < 2:
        return 0.0
    returns = np.diff(np.log(np.clip(p, eps, None)))
    return float(np.sqrt(np.mean(returns**2)))


def build_volatility_targets(
    sequences: np.ndarray, price_index: int = 3, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) pairs for volatility prediction.

    X[i] = sequences[i]                     — current window (input)
    y[i] = realized_volatility(sequences[i+horizon, :, price_index])
           — realised volatility of the close-price path in the next window
    """
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, seq_len, features], got {arr.shape}")
    if len(arr) <= horizon:
        raise ValueError("Not enough sequences for requested horizon")

    X = arr[:-horizon]
    future = arr[horizon:, :, price_index]  # [N, seq_len] close prices in future window
    y = np.array([realized_volatility(seq) for seq in future], dtype=np.float32)
    return X, y


class VolatilityRegressor(nn.Module):
    """
    Default decoder for volatility prediction (task head).

    Maps aggregator embeddings → scalar predicted volatility.
    No dropout here because volatility targets are already noisy; regularisation
    comes from the shallow architecture rather than explicit dropout.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
