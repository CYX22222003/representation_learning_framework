from __future__ import annotations

import copy
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
BYOLVariantName = Literal["byol_lstm", "byol_transformer"]
BYOL_VARIANTS: tuple[BYOLVariantName, ...] = ("byol_lstm", "byol_transformer")


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


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TemporalBYOLEncoder(nn.Module):
    """BYOL online/EMA-target contract with a temporal sequence backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        predictor_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or projection_dim <= 0 or predictor_hidden_dim <= 0:
            raise ValueError("BYOL dimensions must be positive")
        if getattr(backbone, "output_dim", None) != hidden_dim:
            raise ValueError("backbone output dimension must match hidden_dim")
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.online_backbone = backbone
        self.online_projector = _make_mlp(hidden_dim, hidden_dim, projection_dim)
        self.online_predictor = _make_mlp(projection_dim, predictor_hidden_dim, projection_dim)
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        self._freeze_target()

    def _freeze_target(self) -> None:
        for module in (self.target_backbone, self.target_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward_online(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.online_backbone(x)
        z = F.normalize(self.online_projector(h), dim=-1)
        return h, z, self.online_predictor(z)

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_projector(self.target_backbone(x)), dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_backbone(x)

    @torch.no_grad()
    def update_target(self, tau: float = 0.99) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        for online_module, target_module in (
            (self.online_backbone, self.target_backbone),
            (self.online_projector, self.target_projector),
        ):
            for online_parameter, target_parameter in zip(
                online_module.parameters(), target_module.parameters()
            ):
                target_parameter.data.mul_(tau).add_(online_parameter.data, alpha=1.0 - tau)

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1, _, p1 = self.forward_online(view1)
        h2, _, p2 = self.forward_online(view2)
        return h1, h2, p1, p2, self.forward_target(view1), self.forward_target(view2)


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


def _build_temporal_backbone(
    variant: str,
    input_dim: int,
    config: TemporalBackboneConfig,
) -> nn.Module:
    if variant.endswith("_lstm"):
        return SequenceLSTMBackbone(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )
    return SequenceTransformerBackbone(
        input_dim=input_dim,
        model_dim=config.hidden_dim,
        num_layers=config.transformer_num_layers,
        num_heads=config.transformer_num_heads,
        feedforward_dim=config.transformer_feedforward_dim,
        dropout=config.transformer_dropout,
        norm_first=config.transformer_norm_first,
        max_seq_len=config.max_seq_len,
    )


def build_byol_variant(
    variant: str,
    input_dim: int,
    projection_dim: int = 128,
    predictor_hidden_dim: int = 128,
    backbone_config: TemporalBackboneConfig | None = None,
) -> TemporalBYOLEncoder:
    """Construct a named temporal BYOL backbone while preserving BYOL semantics."""
    if variant not in BYOL_VARIANTS:
        raise ValueError(f"unknown BYOL variant {variant!r}; expected one of {BYOL_VARIANTS}")
    config = backbone_config or TemporalBackboneConfig()
    return TemporalBYOLEncoder(
        backbone=_build_temporal_backbone(variant, input_dim, config),
        hidden_dim=config.hidden_dim,
        projection_dim=projection_dim,
        predictor_hidden_dim=predictor_hidden_dim,
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
