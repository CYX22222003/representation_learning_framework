"""Instance normalization, channelwise patching, and fixed positions."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import TimeDARTConfig


def normalize_instance(
    x: torch.Tensor, *, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize each channel within each row; preserve detached scale metadata."""
    mean = x.mean(dim=1, keepdim=True).detach()
    std = (x.var(dim=1, keepdim=True, unbiased=False) + epsilon).sqrt().detach()
    return (x - mean) / std, mean, std


def channelwise_patches(x: torch.Tensor, config: TimeDARTConfig) -> torch.Tensor:
    """Map [batch, time, channel] to [batch*channel, patch, point]."""
    if x.ndim != 3 or tuple(x.shape[1:]) != (config.input_len, config.channels):
        raise ValueError("TimeDART input shape differs from the declared length/channels")
    batch = x.shape[0]
    if batch == 0 or not torch.isfinite(x).all():
        raise ValueError("TimeDART requires a nonempty finite input")
    series = x.transpose(1, 2).reshape(batch * config.channels, config.input_len)
    return series.unfold(-1, config.patch_len, config.stride)


def unpatch_channels(patches: torch.Tensor, config: TimeDARTConfig) -> torch.Tensor:
    if (
        patches.ndim != 3
        or patches.shape[0] % config.channels
        or tuple(patches.shape[1:]) != (config.patch_count, config.patch_len)
    ):
        raise ValueError("TimeDART reconstruction patches have the wrong shape")
    batch = patches.shape[0] // config.channels
    return patches.reshape(batch, config.channels, config.input_len).transpose(1, 2)


class SinusoidalPositions(nn.Module):
    def __init__(self, *, length: int, width: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        angles = positions * frequencies
        encoding = torch.zeros(length, width, dtype=torch.float32)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles.cos()[:, : encoding[:, 1::2].shape[1]]
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        if embedded.ndim != 3 or embedded.shape[1] > self.encoding.shape[1]:
            raise ValueError("position encoding received an invalid patch sequence")
        return self.dropout(
            embedded + self.encoding[:, : embedded.shape[1]].to(dtype=embedded.dtype)
        )
