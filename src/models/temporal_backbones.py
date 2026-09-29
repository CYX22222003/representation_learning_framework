from __future__ import annotations

import math

import torch
import torch.nn as nn


def _validate_sequence(x: torch.Tensor, input_dim: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"expected input shaped [batch, time, features], got {tuple(x.shape)}")
    if x.shape[1] < 1:
        raise ValueError("sequence length must be positive")
    if x.shape[2] != input_dim:
        raise ValueError(f"expected {input_dim} input features, got {x.shape[2]}")


class SequenceLSTMBackbone(nn.Module):
    """Compact recurrent backbone returning one representation per sequence."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("input_dim, hidden_dim, and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence(x, self.input_dim)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class PreActivationResidualBlock1D(nn.Module):
    """Fixed-width pre-activation residual block for temporal sequences."""

    def __init__(
        self,
        channels: int = 128,
        groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups != 0:
            raise ValueError("channels must be positive and divisible by groups")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.channels = channels
        self.groups = groups
        self.dropout_probability = dropout
        self.norm1 = nn.GroupNorm(groups, channels)
        self.activation1 = nn.GELU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation2 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        value = self.conv1(self.activation1(self.norm1(x)))
        value = self.conv2(self.dropout(self.activation2(self.norm2(value))))
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"expected [batch,{self.channels},time], got {tuple(x.shape)}"
            )
        return x + self.residual(x)


class SequenceResidualCNNBackbone(nn.Module):
    """Phase 6.5D fixed three-block residual temporal CNN backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        groups: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_blocks <= 0:
            raise ValueError("input_dim, hidden_dim, and num_blocks must be positive")
        if hidden_dim % groups != 0:
            raise ValueError("groups must divide hidden_dim")
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.num_blocks = num_blocks
        self.groups = groups
        self.dropout = dropout
        self.stem = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            [
                PreActivationResidualBlock1D(hidden_dim, groups, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence(x, self.input_dim)
        value = self.stem(x.transpose(1, 2))
        for block in self.blocks:
            value = block(value)
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward_features(x)).squeeze(-1)


class SinusoidalPositionEncoding(nn.Module):
    """Deterministic positional encoding for batch-first sequence tensors."""

    def __init__(self, model_dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        if model_dim <= 0 or max_seq_len <= 0:
            raise ValueError("model_dim and max_seq_len must be positive")
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / model_dim)
        )
        encoding = torch.zeros(max_seq_len, model_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if model_dim > 1:
            encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input shaped [batch, time, features], got {tuple(x.shape)}")
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"sequence length {x.shape[1]} exceeds configured maximum {self.encoding.shape[1]}"
            )
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class SequenceTransformerBackbone(nn.Module):
    """Compact Transformer encoder with final-token sequence readout."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        norm_first: bool = False,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or model_dim <= 0 or num_layers <= 0:
            raise ValueError("input_dim, model_dim, and num_layers must be positive")
        if num_heads <= 0 or model_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide model_dim")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.output_dim = model_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_encoding = SinusoidalPositionEncoding(model_dim, max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_sequence(x, self.input_dim)
        encoded = self.position_encoding(self.input_projection(x))
        encoded = self.encoder(encoded)
        return self.output_norm(encoded[:, -1])
