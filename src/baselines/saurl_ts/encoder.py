from __future__ import annotations

import torch
import torch.nn as nn


class DilatedResidualBlock(nn.Module):
    """Two-convolution residual block used by both SaDA and SaSSL."""

    def __init__(
        self,
        width: int,
        *,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if width <= 0 or dilation <= 0:
            raise ValueError("width and dilation must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout1(self.activation(self.conv1(x)))
        x = self.dropout2(self.conv2(x))
        return self.activation(residual + x)


class DilatedCNNEncoder(nn.Module):
    """Selected SaURL branch encoder with global-maximum row pooling."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        output_dim: int = 128,
        *,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(
                hidden_dim,
                dilation=dilation,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for dilation in dilations
        )
        self.output_projection = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward_states(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [batch, time, {self.input_dim}], got {tuple(x.shape)}"
            )
        states = self.input_projection(x.transpose(1, 2))
        for block in self.blocks:
            states = block(states)
        return self.output_projection(states).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_states(x).amax(dim=1)
