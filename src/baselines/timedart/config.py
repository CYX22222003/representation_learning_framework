"""Frozen original-TimeDART architecture settings for the project adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TimeDARTConfig:
    input_len: int = 64
    channels: int = 5
    patch_len: int = 2
    stride: int = 2
    d_model: int = 32
    n_heads: int = 8
    d_ff: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 1
    dropout: float = 0.2
    projector_dropout: float = 0.1
    diffusion_steps: int = 1000
    instance_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.input_len <= 0 or self.channels <= 0 or self.patch_len <= 0:
            raise ValueError("input length, channels, and patch length must be positive")
        if self.stride != self.patch_len or self.input_len % self.patch_len:
            raise ValueError("TimeDART requires complete non-overlapping patches")
        if self.d_model <= 0 or self.n_heads <= 0 or self.d_model % self.n_heads:
            raise ValueError("model width must be positive and divisible by attention heads")
        if self.d_ff <= 0 or self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError("feed-forward width and layer counts must be positive")
        if self.diffusion_steps <= 1 or self.instance_norm_epsilon <= 0:
            raise ValueError("diffusion steps and normalization epsilon are invalid")
        if not 0 <= self.dropout < 1 or not 0 <= self.projector_dropout < 1:
            raise ValueError("dropout probabilities must be in [0, 1)")

    @property
    def patch_count(self) -> int:
        return self.input_len // self.patch_len

    @property
    def representation_width(self) -> int:
        # Five channelwise pooled vectors plus per-window mean and std.
        return self.channels * (self.d_model + 2)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)
