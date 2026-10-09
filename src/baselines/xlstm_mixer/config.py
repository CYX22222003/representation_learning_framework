"""Frozen Phase 6.9 configuration and provenance for xLSTM-Mixer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any


METHOD_ID = "XM-MV8"
CHECKPOINT_SCHEMA = "phase6-9-xlstm-mixer-model-v1"


@dataclass(frozen=True)
class XLSTMMixerSourceContract:
    """Pinned scientific and behavioral sources for the project adapter."""

    official_repository: str = "https://github.com/mauricekraus/xlstm-mixer"
    official_source_commit: str = "730b0531aa9456e498765028f3c22ca3677de42e"
    official_model_sha256: str = (
        "d87a41bb96d86baea1be240b8244934426bcbd9fb519b7468a541ebf2c226bbe"
    )
    official_wrapper_license: str = "MIT"
    xlstm_version: str = "1.0.3"
    xlstm_repository: str = "https://github.com/NX-AI/xlstm"
    xlstm_commit: str = "1ff240242795062e56b4e39b43023cce61e8e88c"
    xlstm_license: str = "AGPL-3.0"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


SOURCE_CONTRACT = XLSTMMixerSourceContract()


@dataclass(frozen=True)
class XLSTMMixerConfig:
    """Architecture contract.

    Every field is the owner-approved ``XM-MV8`` configuration. Architecture
    variants require a dated amendment and cannot silently retain this method
    identity.
    """

    context_length: int = 64
    horizon: int = 8
    num_variates: int = 5
    embedding_dim: int = 128
    num_heads: int = 8
    num_blocks: int = 1
    dropout: float = 0.1
    conv1d_kernel_size: int = 0
    num_initial_tokens: int = 1
    num_tokens_per_variate: int = 1
    packing: int = 1
    revin_epsilon: float = 1e-5
    channel_order: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    backbone: str = "nlinear"
    reverse_view: str = "latent_feature_axis"
    revin_affine: bool = False
    full_path_loss: str = "l1_mean"

    def __post_init__(self) -> None:
        positive = {
            "context_length": self.context_length,
            "horizon": self.horizon,
            "num_variates": self.num_variates,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"positive xLSTM-Mixer fields required: {invalid}")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.revin_epsilon <= 0:
            raise ValueError("revin_epsilon must be positive")
        if len(self.channel_order) != self.num_variates:
            raise ValueError("channel_order length must equal num_variates")
        if len(set(self.channel_order)) != len(self.channel_order):
            raise ValueError("channel_order entries must be unique")
        frozen = {
            "context_length": (self.context_length, 64),
            "horizon": (self.horizon, 8),
            "num_variates": (self.num_variates, 5),
            "embedding_dim": (self.embedding_dim, 128),
            "num_heads": (self.num_heads, 8),
            "num_blocks": (self.num_blocks, 1),
            "dropout": (self.dropout, 0.1),
            "conv1d_kernel_size": (self.conv1d_kernel_size, 0),
            "num_initial_tokens": (self.num_initial_tokens, 1),
            "num_tokens_per_variate": (self.num_tokens_per_variate, 1),
            "packing": (self.packing, 1),
            "backbone": (self.backbone, "nlinear"),
            "reverse_view": (self.reverse_view, "latent_feature_axis"),
            "revin_affine": (self.revin_affine, False),
            "full_path_loss": (self.full_path_loss, "l1_mean"),
            "channel_order": (
                self.channel_order,
                ("open", "high", "low", "close", "volume"),
            ),
        }
        changed = [name for name, (actual, expected) in frozen.items() if actual != expected]
        if changed:
            raise ValueError(f"frozen XM-MV8 fields changed: {changed}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channel_order"] = list(self.channel_order)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "XLSTMMixerConfig":
        values = dict(payload)
        if "channel_order" in values:
            values["channel_order"] = tuple(values["channel_order"])
        return cls(**values)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


DEFAULT_XLSTM_MIXER_CONFIG = XLSTMMixerConfig()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()
