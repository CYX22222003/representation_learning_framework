"""Neural building blocks for the independent LWA reconstruction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_padding(length: int, kernel_size: int, stride: int) -> tuple[int, int]:
    output_length = math.ceil(length / stride)
    total = max(0, (output_length - 1) * stride + kernel_size - length)
    return total // 2, total - total // 2


class SamePadConv1d(nn.Module):
    """Conv1d with TensorFlow-style SAME output length semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or stride <= 0:
            raise ValueError("kernel_size and stride must be positive")
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left, right = _same_padding(inputs.shape[-1], self.kernel_size, self.stride)
        return self.conv(F.pad(inputs, (left, right)))


class SamePadMaxPool1d(nn.Module):
    def __init__(self, kernel_size: int, *, stride: int | None = None) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left, right = _same_padding(inputs.shape[-1], self.kernel_size, self.stride)
        # Zero padding is part of the audited source-style residual shortcut.
        return F.max_pool1d(
            F.pad(inputs, (left, right)),
            kernel_size=self.kernel_size,
            stride=self.stride,
        )


class ResidualBlock1D(nn.Module):
    """Pre-activation residual block with optional stride and channel growth."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        first_block: bool = False,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_channels < in_channels:
            raise ValueError("ResidualBlock1D cannot shrink the channel dimension")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.first_block = first_block
        self.norm1 = (
            nn.BatchNorm1d(in_channels)
            if use_batch_norm and not first_block
            else nn.Identity()
        )
        self.norm2 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.dropout1 = (
            nn.Dropout(dropout) if dropout > 0.0 and not first_block else nn.Identity()
        )
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv1 = SamePadConv1d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.conv2 = SamePadConv1d(out_channels, out_channels, kernel_size)
        self.shortcut_pool = (
            SamePadMaxPool1d(stride, stride=stride) if stride > 1 else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut_pool(inputs)
        hidden = inputs
        if not self.first_block:
            hidden = self.dropout1(F.relu(self.norm1(hidden)))
        hidden = self.conv1(hidden)
        hidden = self.conv2(self.dropout2(F.relu(self.norm2(hidden))))

        if self.out_channels != self.in_channels:
            before = (self.out_channels - self.in_channels) // 2
            after = self.out_channels - self.in_channels - before
            identity = F.pad(identity.transpose(1, 2), (before, after)).transpose(1, 2)
        if hidden.shape != identity.shape:
            raise RuntimeError(
                "residual and shortcut shapes differ: "
                f"{tuple(hidden.shape)} vs {tuple(identity.shape)}"
            )
        return hidden + identity


class ConvNormReLU2d(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
