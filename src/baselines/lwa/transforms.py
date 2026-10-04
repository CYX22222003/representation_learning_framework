"""Deterministic LWA frame projections.

This is an independently authored, paper-guided implementation. The upstream
repository was used only to audit behavior and was not copied because it has no
software licence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import LWAConfig


def validate_time_batch(batch: torch.Tensor, config: LWAConfig) -> None:
    expected_tail = (config.sequence_length, config.input_channels)
    if batch.ndim != 3 or tuple(batch.shape[1:]) != expected_tail:
        raise ValueError(
            f"expected [batch, {config.sequence_length}, {config.input_channels}], "
            f"got {tuple(batch.shape)}"
        )
    if not batch.is_floating_point() or batch.is_complex():
        raise TypeError("LWA time inputs must be real floating-point tensors")
    if not torch.isfinite(batch).all():
        raise ValueError("LWA time inputs must be finite")


def orthonormal_rfft(
    batch: torch.Tensor, config: LWAConfig | None = None
) -> torch.Tensor:
    """Transform the time axis of ``[B, T, C]`` into complex ``[B, C, F]``."""

    resolved = config or LWAConfig()
    validate_time_batch(batch, resolved)
    transformed = torch.fft.rfft(batch.transpose(1, 2), dim=-1, norm="ortho")
    expected = (batch.shape[0], resolved.input_channels, resolved.frequency_bins)
    if tuple(transformed.shape) != expected:
        raise RuntimeError(f"unexpected rFFT shape {tuple(transformed.shape)}")
    return transformed


def _load_pywavelets():
    try:
        import pywt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyWavelets is required for LWA CWT construction; install the "
            "project-pinned PyWavelets==1.8.0"
        ) from exc
    return pywt


@dataclass(frozen=True)
class MorletCWT:
    """CPU-only channelwise complex-Morlet CWT magnitude transform."""

    config: LWAConfig = LWAConfig()

    @property
    def scales(self) -> np.ndarray:
        return np.asarray(self.config.wavelet_scales, dtype=np.float64)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        validate_time_batch(batch, self.config)
        pywt = _load_pywavelets()
        source = batch.detach().to(device="cpu", dtype=torch.float32).numpy()
        output = np.empty(
            (
                source.shape[0],
                self.config.input_channels,
                self.config.wavelet_scale_count,
                self.config.sequence_length,
            ),
            dtype=np.float32,
        )
        sampling_period = 1.0 / self.config.sampling_frequency_per_hour
        for sample_index in range(source.shape[0]):
            for channel_index in range(self.config.input_channels):
                coefficients, _ = pywt.cwt(
                    source[sample_index, :, channel_index],
                    self.scales,
                    self.config.wavelet_name,
                    sampling_period=sampling_period,
                )
                output[sample_index, channel_index] = np.abs(coefficients).astype(
                    np.float32, copy=False
                )
        result = torch.from_numpy(output)
        if not torch.isfinite(result).all():
            raise ValueError("LWA CWT output contains non-finite values")
        return result
