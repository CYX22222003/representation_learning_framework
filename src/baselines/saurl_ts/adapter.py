from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .config import SaURLConfig
from .model import SaURLModel, build_saurl


@dataclass(frozen=True)
class SaURLInputScaler:
    """Five-channel training-population scaler used before the SaURL model."""

    mean: torch.Tensor
    scale: torch.Tensor
    minimum_scale: float = 1e-8

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.scale.ndim != 1:
            raise ValueError("mean and scale must be one-dimensional")
        if self.mean.shape != self.scale.shape:
            raise ValueError("mean and scale must have identical shapes")
        if not self.mean.is_floating_point() or not self.scale.is_floating_point():
            raise TypeError("mean and scale must be floating point")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all():
            raise ValueError("scaler statistics must be finite")
        if not torch.all(self.scale > 0):
            raise ValueError("all scaler values must be positive")

    @classmethod
    def fit(cls, sequences: torch.Tensor, minimum_scale: float = 1e-8) -> SaURLInputScaler:
        if sequences.ndim != 3 or sequences.shape[0] == 0 or sequences.shape[1] == 0:
            raise ValueError("sequences must be a non-empty [rows, time, channels] tensor")
        if not sequences.is_floating_point() or not torch.isfinite(sequences).all():
            raise ValueError("sequences must be finite floating-point values")
        if minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        mean = sequences.mean(dim=(0, 1)).detach().cpu()
        standard_deviation = sequences.std(dim=(0, 1), unbiased=False).detach().cpu()
        scale = torch.where(
            standard_deviation < minimum_scale,
            torch.ones_like(standard_deviation),
            standard_deviation,
        )
        return cls(mean=mean, scale=scale, minimum_scale=minimum_scale)

    def transform(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3 or sequences.shape[-1] != self.mean.numel():
            raise ValueError(
                f"expected [rows, time, {self.mean.numel()}], got {tuple(sequences.shape)}"
            )
        mean = self.mean.to(device=sequences.device, dtype=sequences.dtype)
        scale = self.scale.to(device=sequences.device, dtype=sequences.dtype)
        return (sequences - mean) / scale

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "mean": self.mean.detach().cpu().clone(),
            "scale": self.scale.detach().cpu().clone(),
            "minimum_scale": self.minimum_scale,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, torch.Tensor | float]
    ) -> SaURLInputScaler:
        mean = state.get("mean")
        scale = state.get("scale")
        minimum_scale = state.get("minimum_scale", 1e-8)
        if not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor):
            raise TypeError("scaler state must contain tensor mean and scale")
        if not isinstance(minimum_scale, (int, float)):
            raise TypeError("minimum_scale must be numeric")
        return cls(mean=mean.clone(), scale=scale.clone(), minimum_scale=float(minimum_scale))


class SaURLAdapter:
    """Project-facing normalization and frozen-extraction boundary."""

    def __init__(self, model: SaURLModel, scaler: SaURLInputScaler) -> None:
        if scaler.mean.numel() != model.config.input_dim:
            raise ValueError("scaler width does not match model input_dim")
        self.model = model
        self.scaler = scaler

    def normalize(self, raw_batch: torch.Tensor) -> torch.Tensor:
        normalized = self.scaler.transform(raw_batch)
        self.model._validate_batch(normalized)
        return normalized

    @torch.no_grad()
    def encode(self, raw_batch: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.encode(self.normalize(raw_batch))


def build_saurl_adapter(
    scaler: SaURLInputScaler,
    config: SaURLConfig | None = None,
) -> SaURLAdapter:
    return SaURLAdapter(build_saurl(config), scaler)
