"""Domain encoders for the clean-room, paper-guided LWA implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormReLU2d, ResidualBlock1D, SamePadConv1d
from .config import LWAConfig
from .transforms import validate_time_batch


class TimeDomainEncoder(nn.Module):
    def __init__(self, config: LWAConfig | None = None) -> None:
        super().__init__()
        self.config = config or LWAConfig()
        self.stem = SamePadConv1d(
            self.config.input_channels,
            self.config.time_base_channels,
            self.config.time_kernel_size,
        )
        self.stem_norm = nn.BatchNorm1d(self.config.time_base_channels)

        blocks: list[nn.Module] = []
        current_channels = self.config.time_base_channels
        for index in range(self.config.time_block_count):
            block_number = index + 1
            output_channels = current_channels
            if index > 0 and index % self.config.time_channel_increase_gap == 0:
                output_channels *= 2
            stride = (
                self.config.time_downsample_stride
                if block_number % self.config.time_downsample_gap == 0
                else 1
            )
            blocks.append(
                ResidualBlock1D(
                    current_channels,
                    output_channels,
                    kernel_size=self.config.time_kernel_size,
                    stride=stride,
                    first_block=index == 0,
                    use_batch_norm=True,
                    dropout=self.config.time_dropout,
                )
            )
            current_channels = output_channels
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.BatchNorm1d(current_channels)
        self.output = nn.Linear(current_channels, self.config.representation_dim)

    def forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        validate_time_batch(batch, self.config)
        hidden = F.relu(self.stem_norm(self.stem(batch.transpose(1, 2))))
        for block in self.blocks:
            hidden = block(hidden)
        return F.relu(self.final_norm(hidden))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_features(batch).mean(dim=-1)
        return self.output(hidden)


class _FourierBranch(nn.Module):
    def __init__(self, config: LWAConfig) -> None:
        super().__init__()
        first_width, second_width = config.fourier_branch_channels
        self.stem = nn.Conv1d(
            config.input_channels,
            config.fourier_stem_channels,
            kernel_size=config.fourier_kernel_size,
            padding=config.fourier_kernel_size // 2,
        )
        self.blocks = nn.ModuleList(
            (
                ResidualBlock1D(
                    config.fourier_stem_channels,
                    first_width,
                    kernel_size=config.fourier_kernel_size,
                    stride=2,
                    first_block=True,
                    use_batch_norm=False,
                ),
                ResidualBlock1D(
                    first_width,
                    second_width,
                    kernel_size=config.fourier_kernel_size,
                    stride=2,
                    use_batch_norm=False,
                ),
            )
        )
        self.reduce_frequency = nn.Linear(config.fourier_reduced_length, 1)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.stem(inputs))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.reduce_frequency(self.forward_features(inputs)).squeeze(-1)


class FourierDomainEncoder(nn.Module):
    def __init__(self, config: LWAConfig | None = None) -> None:
        super().__init__()
        self.config = config or LWAConfig()
        self.magnitude_branch = _FourierBranch(self.config)
        self.phase_branch = _FourierBranch(self.config)

    def _validate(self, transformed: torch.Tensor) -> None:
        expected = (
            self.config.input_channels,
            self.config.frequency_bins,
        )
        if transformed.ndim != 3 or tuple(transformed.shape[1:]) != expected:
            raise ValueError(
                f"expected complex [batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(transformed.shape)}"
            )
        if not transformed.is_complex():
            raise TypeError("the Fourier encoder requires a complex rFFT tensor")
        if not torch.isfinite(transformed.real).all() or not torch.isfinite(
            transformed.imag
        ).all():
            raise ValueError("Fourier inputs must be finite")

    def forward_branch_features(
        self, transformed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(transformed)
        magnitude = self.magnitude_branch.forward_features(transformed.abs().float())
        phase = self.phase_branch.forward_features(torch.angle(transformed).float())
        return magnitude, phase

    def forward(self, transformed: torch.Tensor) -> torch.Tensor:
        self._validate(transformed)
        magnitude = self.magnitude_branch(transformed.abs().float())
        phase = self.phase_branch(torch.angle(transformed).float())
        return torch.cat((magnitude, phase), dim=1)


class WaveletDomainEncoder(nn.Module):
    def __init__(self, config: LWAConfig | None = None) -> None:
        super().__init__()
        self.config = config or LWAConfig()
        base = self.config.wavelet_base_channels
        channels = self.config.input_channels
        kernel = self.config.wavelet_kernel_size
        self.input_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.input_pool_4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.layer1 = ConvNormReLU2d(
            channels, base, kernel_size=kernel, stride=1
        )
        self.layer2 = ConvNormReLU2d(
            base, 2 * base, kernel_size=kernel, stride=2
        )
        self.layer3 = ConvNormReLU2d(
            2 * base + channels, 3 * base, kernel_size=kernel, stride=2
        )
        self.layer4 = ConvNormReLU2d(
            3 * base + channels, 4 * base, kernel_size=kernel, stride=2
        )
        reduced_scales, reduced_time = self.config.wavelet_reduced_shape
        self.reduce_time = nn.Linear(reduced_time, 1)
        self.reduce_scales = nn.Linear(reduced_scales, 1)

    def _validate(self, transformed: torch.Tensor) -> None:
        expected = (
            self.config.input_channels,
            self.config.wavelet_scale_count,
            self.config.sequence_length,
        )
        if transformed.ndim != 4 or tuple(transformed.shape[1:]) != expected:
            raise ValueError(
                f"expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(transformed.shape)}"
            )
        if not transformed.is_floating_point() or transformed.is_complex():
            raise TypeError("the wavelet encoder requires real CWT magnitudes")
        if not torch.isfinite(transformed).all():
            raise ValueError("wavelet inputs must be finite")

    def forward_features(self, transformed: torch.Tensor) -> torch.Tensor:
        self._validate(transformed)
        pooled_2 = self.input_pool_2(transformed)
        pooled_4 = self.input_pool_4(transformed)
        hidden = self.layer1(transformed)
        hidden = self.layer2(hidden)
        hidden = self.layer3(torch.cat((hidden, pooled_2), dim=1))
        hidden = self.layer4(torch.cat((hidden, pooled_4), dim=1))
        expected = (
            transformed.shape[0],
            self.config.representation_dim,
            *self.config.wavelet_reduced_shape,
        )
        if tuple(hidden.shape) != expected:
            raise RuntimeError(
                f"unexpected wavelet feature shape {tuple(hidden.shape)}"
            )
        return hidden

    def forward(self, transformed: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_features(transformed)
        hidden = self.reduce_time(hidden).squeeze(-1)
        return self.reduce_scales(hidden).squeeze(-1)
