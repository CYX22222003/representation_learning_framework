from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SaURLConfig:
    """Frozen Phase 6.7 SaURL-TS reconstruction contract."""

    sequence_length: int = 64
    input_dim: int = 5
    augmentation_hidden_dim: int = 16
    encoder_hidden_dim: int = 64
    representation_dim: int = 128
    projector_hidden_dim: int = 128
    projection_dim: int = 128
    predictor_hidden_dim: int = 128
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout: float = 0.1
    attention_regions: int = 8
    mask_threshold: float = 0.5
    sada_alpha: float = 0.1
    sada_beta: float = 0.01
    sada_gamma: float = 0.5
    diversity_weight: float = 1.25
    ema_tau: float = 0.99
    sada_learning_rate: float = 1e-2
    sassl_learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    ratio_step: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "dilations", tuple(self.dilations))
        positive = {
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "augmentation_hidden_dim": self.augmentation_hidden_dim,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "representation_dim": self.representation_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
            "projection_dim": self.projection_dim,
            "predictor_hidden_dim": self.predictor_hidden_dim,
            "kernel_size": self.kernel_size,
            "attention_regions": self.attention_regions,
            "batch_size": self.batch_size,
            "ratio_step": self.ratio_step,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.dilations or any(dilation <= 0 for dilation in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so residual lengths are preserved")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.representation_dim % self.attention_regions != 0:
            raise ValueError("attention_regions must divide representation_dim")
        if not 0.0 < self.mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")
        for name in ("sada_alpha", "sada_beta", "sada_gamma", "diversity_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.ema_tau <= 1.0:
            raise ValueError("ema_tau must be in [0, 1]")
        if self.sada_learning_rate <= 0.0 or self.sassl_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")

    @property
    def frequency_bins(self) -> int:
        return self.sequence_length // 2 + 1

    @property
    def attention_region_width(self) -> int:
        return self.representation_dim // self.attention_regions

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["dilations"] = list(self.dilations)
        return values


DEFAULT_SAURL_CONFIG = SaURLConfig()
