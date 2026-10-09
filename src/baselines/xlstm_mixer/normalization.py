"""Source-aligned non-affine reversible instance normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RevINStatistics:
    mean: torch.Tensor
    standard_deviation: torch.Tensor


class NonAffineRevIN(nn.Module):
    """RevIN with detached per-sample statistics and no learned affine terms."""

    def __init__(self, num_variates: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        if num_variates <= 0 or epsilon <= 0:
            raise ValueError("num_variates and epsilon must be positive")
        self.num_variates = int(num_variates)
        self.epsilon = float(epsilon)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, RevINStatistics]:
        self._validate(x)
        mean = x.mean(dim=1, keepdim=True).detach()
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        standard_deviation = (variance + self.epsilon).sqrt().detach()
        return (x - mean) / standard_deviation, RevINStatistics(mean, standard_deviation)

    def denormalize(
        self, normalized: torch.Tensor, statistics: RevINStatistics
    ) -> torch.Tensor:
        if normalized.ndim != 3 or normalized.shape[-1] != self.num_variates:
            raise ValueError("RevIN denormalization expects [batch, time, variates]")
        expected = (normalized.shape[0], 1, self.num_variates)
        if tuple(statistics.mean.shape) != expected:
            raise ValueError("RevIN mean has the wrong shape")
        if tuple(statistics.standard_deviation.shape) != expected:
            raise ValueError("RevIN standard deviation has the wrong shape")
        return normalized * statistics.standard_deviation + statistics.mean

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.num_variates:
            raise ValueError("RevIN normalization expects [batch, time, variates]")
        if not x.is_floating_point():
            raise TypeError("xLSTM-Mixer inputs must use a floating dtype")
        if not torch.isfinite(x).all():
            raise ValueError("xLSTM-Mixer inputs must be finite")
