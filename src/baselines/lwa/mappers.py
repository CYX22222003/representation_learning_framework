from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )


class ConvolutionalMapper(nn.Module):
    """Map one latent vector through a lightweight 1 -> H -> 1 convolution."""

    def __init__(
        self, vector_dim: int = 128, hidden_channels: int = 64, kernel_size: int = 3
    ) -> None:
        super().__init__()
        if vector_dim <= 0 or vector_dim % 2 != 0:
            raise ValueError("vector_dim must be a positive even number")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        self.vector_dim = vector_dim
        padding = kernel_size // 2
        self.encode = nn.Conv1d(
            1,
            hidden_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
        )
        self.decode = nn.ConvTranspose1d(
            hidden_channels,
            1,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.vector_dim:
            raise ValueError(
                f"expected [batch, {self.vector_dim}], got {tuple(inputs.shape)}"
            )
        output = self.decode(torch.relu(self.encode(inputs.unsqueeze(1)))).squeeze(1)
        if output.shape != inputs.shape:
            raise RuntimeError(
                f"mapper changed latent shape from {tuple(inputs.shape)} "
                f"to {tuple(output.shape)}"
            )
        return output


class MapperPair(nn.Module):
    def __init__(
        self, vector_dim: int = 128, hidden_channels: int = 64, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.fourier = ConvolutionalMapper(
            vector_dim, hidden_channels, kernel_size
        )
        self.wavelet = ConvolutionalMapper(
            vector_dim, hidden_channels, kernel_size
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fourier(inputs), self.wavelet(inputs)
