from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temporal_backbones import SequenceLSTMBackbone, SequenceTransformerBackbone


ContrastiveVariantName = Literal["contrastive_lstm", "contrastive_transformer"]
CONTRASTIVE_VARIANTS: tuple[ContrastiveVariantName, ...] = (
    "contrastive_lstm",
    "contrastive_transformer",
)


@dataclass(frozen=True)
class TemporalBackboneConfig:
    hidden_dim: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_feedforward_dim: int = 256
    transformer_dropout: float = 0.1
    transformer_norm_first: bool = False
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.lstm_num_layers <= 0:
            raise ValueError("hidden_dim and lstm_num_layers must be positive")
        if not 0.0 <= self.lstm_dropout < 1.0:
            raise ValueError("lstm_dropout must be in [0, 1)")
        if self.lstm_num_layers == 1 and self.lstm_dropout != 0.0:
            raise ValueError("lstm_dropout must be zero when lstm_num_layers is one")
        if self.transformer_num_layers <= 0 or self.transformer_num_heads <= 0:
            raise ValueError("Transformer layer and head counts must be positive")
        if self.hidden_dim % self.transformer_num_heads != 0:
            raise ValueError("transformer_num_heads must divide hidden_dim")
        if self.transformer_feedforward_dim <= 0 or self.max_seq_len <= 0:
            raise ValueError("transformer_feedforward_dim and max_seq_len must be positive")
        if not 0.0 <= self.transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TemporalContrastiveEncoder(nn.Module):
    """Temporal backbone plus the Phase-1 contrastive projector contract."""

    def __init__(self, backbone: nn.Module, hidden_dim: int = 128, embedding_dim: int = 128) -> None:
        super().__init__()
        if hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("hidden_dim and embedding_dim must be positive")
        backbone_output_dim = getattr(backbone, "output_dim", None)
        if backbone_output_dim != hidden_dim:
            raise ValueError(
                f"backbone output_dim={backbone_output_dim!r} does not match hidden_dim={hidden_dim}"
            )
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unnormalised 128-d backbone state used downstream."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        z = F.normalize(self.projector(h), dim=-1)
        return h, z


def build_contrastive_variant(
    variant: str,
    input_dim: int,
    embedding_dim: int = 128,
    backbone_config: TemporalBackboneConfig | None = None,
) -> TemporalContrastiveEncoder:
    """Construct a named Phase-2 contrastive backbone substitution."""
    if variant not in CONTRASTIVE_VARIANTS:
        raise ValueError(f"unknown contrastive variant {variant!r}; expected one of {CONTRASTIVE_VARIANTS}")
    config = backbone_config or TemporalBackboneConfig()
    if variant == "contrastive_lstm":
        backbone = SequenceLSTMBackbone(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )
    else:
        backbone = SequenceTransformerBackbone(
            input_dim=input_dim,
            model_dim=config.hidden_dim,
            num_layers=config.transformer_num_layers,
            num_heads=config.transformer_num_heads,
            feedforward_dim=config.transformer_feedforward_dim,
            dropout=config.transformer_dropout,
            norm_first=config.transformer_norm_first,
            max_seq_len=config.max_seq_len,
        )
    return TemporalContrastiveEncoder(
        backbone=backbone,
        hidden_dim=config.hidden_dim,
        embedding_dim=embedding_dim,
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
