from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def realized_volatility(prices: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(prices, dtype=np.float32)
    if len(p) < 2:
        return 0.0
    returns = np.diff(np.log(np.clip(p, eps, None)))
    return float(np.sqrt(np.mean(returns**2)))


def build_volatility_targets(
    sequences: np.ndarray, price_index: int = 3, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [N, seq_len, features], got {arr.shape}")
    if len(arr) <= horizon:
        raise ValueError("Not enough sequences for requested horizon")

    X = arr[:-horizon]
    future = arr[horizon:, :, price_index]
    y = np.array([realized_volatility(seq) for seq in future], dtype=np.float32)
    return X, y


class VolatilityRegressor(nn.Module):
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


def mse_and_corr(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_ = pred.reshape(-1)
    target_ = target.reshape(-1)
    mse = torch.mean((pred_ - target_) ** 2).item()
    corr = float("nan")
    if len(pred_) > 1:
        corr_tensor = torch.corrcoef(torch.stack([pred_, target_]))
        corr = float(corr_tensor[0, 1].item())
    return {"mse": mse, "corr": corr}
