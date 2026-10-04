from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LWAConfig:
    """Frozen Phase 6.7 LWA architecture and objective contract."""

    sequence_length: int = 64
    input_channels: int = 5
    representation_dim: int = 128
    projection_hidden_dim: int = 128
    projection_dim: int = 128

    time_base_channels: int = 32
    time_block_count: int = 8
    time_kernel_size: int = 5
    time_downsample_stride: int = 2
    time_downsample_gap: int = 2
    time_channel_increase_gap: int = 4
    time_dropout: float = 0.5

    fourier_stem_channels: int = 16
    fourier_branch_channels: tuple[int, int] = (32, 64)
    fourier_kernel_size: int = 3

    wavelet_name: str = "cmor1-1"
    wavelet_scale_count: int = 48
    wavelet_min_scale: float = 1.0
    wavelet_max_scale: float = 128.0
    wavelet_base_channels: int = 32
    wavelet_kernel_size: int = 5
    sampling_frequency_per_hour: float = 1.0

    mapper_hidden_channels: int = 64
    mapper_kernel_size: int = 3
    temperature: float = 0.15

    physical_batch_size: int = 128
    joint_epochs: int = 50
    mapper_epochs: int = 50
    learning_rate: float = 3e-3
    weight_decay: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fourier_branch_channels", tuple(self.fourier_branch_channels)
        )
        positive_integer_fields = (
            "sequence_length",
            "input_channels",
            "representation_dim",
            "projection_hidden_dim",
            "projection_dim",
            "time_base_channels",
            "time_block_count",
            "time_kernel_size",
            "time_downsample_stride",
            "time_downsample_gap",
            "time_channel_increase_gap",
            "fourier_stem_channels",
            "fourier_kernel_size",
            "wavelet_scale_count",
            "wavelet_base_channels",
            "wavelet_kernel_size",
            "mapper_hidden_channels",
            "mapper_kernel_size",
            "physical_batch_size",
            "joint_epochs",
            "mapper_epochs",
        )
        for name in positive_integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.fourier_branch_channels) != 2 or any(
            width <= 0 for width in self.fourier_branch_channels
        ):
            raise ValueError("fourier_branch_channels must contain two positive widths")
        for name in (
            "time_kernel_size",
            "fourier_kernel_size",
            "wavelet_kernel_size",
            "mapper_kernel_size",
        ):
            if getattr(self, name) % 2 == 0:
                raise ValueError(f"{name} must be odd")
        if not 0.0 <= self.time_dropout < 1.0:
            raise ValueError("time_dropout must be in [0, 1)")
        if self.wavelet_min_scale <= 0.0:
            raise ValueError("wavelet_min_scale must be positive")
        if self.wavelet_max_scale <= self.wavelet_min_scale:
            raise ValueError("wavelet_max_scale must exceed wavelet_min_scale")
        if self.sampling_frequency_per_hour <= 0.0:
            raise ValueError("sampling_frequency_per_hour must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

        # The transformed-domain reductions are analytically tied to this shape.
        if self.sequence_length != 64 or self.input_channels != 5:
            raise ValueError("the approved LWA adapter supports only [B, 64, 5] input")
        if self.representation_dim != 128 or self.projection_dim != 128:
            raise ValueError(
                "the approved LWA representation and projection widths are 128"
            )
        if self.wavelet_scale_count != 48:
            raise ValueError("the approved LWA CWT uses exactly 48 scales")
        if self.time_downsample_stride != 2:
            raise ValueError("the approved LWA residual downsampling stride is 2")
        if 2 * self.fourier_branch_channels[-1] != self.representation_dim:
            raise ValueError(
                "the two Fourier branches must concatenate to representation_dim"
            )
        if 4 * self.wavelet_base_channels != self.representation_dim:
            raise ValueError(
                "the final wavelet width must equal representation_dim"
            )

    @property
    def frequency_bins(self) -> int:
        return self.sequence_length // 2 + 1

    @property
    def fourier_reduced_length(self) -> int:
        length = self.frequency_bins
        for _ in self.fourier_branch_channels:
            length = math.ceil(length / self.time_downsample_stride)
        return length

    @property
    def wavelet_reduced_shape(self) -> tuple[int, int]:
        scales = self.wavelet_scale_count
        time = self.sequence_length
        for _ in range(3):
            scales = math.ceil(scales / 2)
            time = math.ceil(time / 2)
        return scales, time

    @property
    def inference_dim(self) -> int:
        return 3 * self.representation_dim

    @property
    def wavelet_scales(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in np.geomspace(
                self.wavelet_min_scale,
                self.wavelet_max_scale,
                num=self.wavelet_scale_count,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["fourier_branch_channels"] = list(self.fourier_branch_channels)
        values["wavelet_scales"] = list(self.wavelet_scales)
        values["frequency_bins"] = self.frequency_bins
        values["fourier_reduced_length"] = self.fourier_reduced_length
        values["wavelet_reduced_shape"] = list(self.wavelet_reduced_shape)
        values["inference_dim"] = self.inference_dim
        return values


DEFAULT_LWA_CONFIG = LWAConfig()
