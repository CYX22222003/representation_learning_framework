from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import SaURLConfig
from .encoder import DilatedResidualBlock


@dataclass(frozen=True)
class ViewParts:
    soft_mask: torch.Tensor
    mask: torch.Tensor
    informative: torch.Tensor
    irrelevant: torch.Tensor
    informative_scale: torch.Tensor
    irrelevant_scale: torch.Tensor
    transformed_informative: torch.Tensor
    transformed_irrelevant: torch.Tensor
    view: torch.Tensor


@dataclass(frozen=True)
class DomainAugmentation:
    source: torch.Tensor
    embedding: torch.Tensor
    factor_logits: torch.Tensor
    view1: ViewParts
    view2: ViewParts


@dataclass(frozen=True)
class SaURLViews:
    temporal_view1: torch.Tensor
    temporal_view2: torch.Tensor
    frequency_view1: torch.Tensor
    frequency_view2: torch.Tensor

    def detached(self) -> SaURLViews:
        return SaURLViews(
            temporal_view1=self.temporal_view1.detach(),
            temporal_view2=self.temporal_view2.detach(),
            frequency_view1=self.frequency_view1.detach(),
            frequency_view2=self.frequency_view2.detach(),
        )


@dataclass(frozen=True)
class SaURLViewsAndParts:
    temporal: DomainAugmentation
    frequency: DomainAugmentation
    frequency_phase: torch.Tensor
    views: SaURLViews

    def detached_views(self) -> SaURLViews:
        return self.views.detached()


class AugmentationEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.residual = DilatedResidualBlock(
            hidden_dim, dilation=1, kernel_size=3, dropout=dropout
        )

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim != 3 or source.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected [batch, positions, {self.input_dim}], got {tuple(source.shape)}"
            )
        hidden = self.input_projection(source.transpose(1, 2))
        return self.residual(hidden).transpose(1, 2)


class SelfAdaptiveDataAugmentation(nn.Module):
    """One domain-specific SaDA factorizer with two independent view heads."""

    def __init__(self, config: SaURLConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.augmentation_hidden_dim
        self.encoder = AugmentationEncoder(config.input_dim, hidden, config.dropout)
        self.factor_head = nn.Linear(hidden, 1)
        self.informative_heads = nn.ModuleList((nn.Linear(hidden, 1), nn.Linear(hidden, 1)))
        self.irrelevant_heads = nn.ModuleList((nn.Linear(hidden, 1), nn.Linear(hidden, 1)))

    def _mask(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The paper specifies a deterministic threshold of the factor-head
        # sigmoid. The straight-through form preserves that exact forward value
        # while supplying a usable gradient for the otherwise discrete mask.
        soft = torch.sigmoid(logits)
        hard = (soft > self.config.mask_threshold).to(dtype=soft.dtype)
        straight_through = hard - soft.detach() + soft
        return soft, straight_through

    def _make_view(
        self,
        source: torch.Tensor,
        embedding: torch.Tensor,
        factor_logits: torch.Tensor,
        view_index: int,
    ) -> ViewParts:
        soft_mask, mask = self._mask(factor_logits)
        informative = mask * source
        irrelevant = (1.0 - mask) * source
        informative_scale = torch.sigmoid(self.informative_heads[view_index](embedding))
        irrelevant_scale = torch.sigmoid(self.irrelevant_heads[view_index](embedding))
        transformed_informative = informative_scale * informative
        transformed_irrelevant = irrelevant_scale * irrelevant
        return ViewParts(
            soft_mask=soft_mask,
            mask=mask,
            informative=informative,
            irrelevant=irrelevant,
            informative_scale=informative_scale,
            irrelevant_scale=irrelevant_scale,
            transformed_informative=transformed_informative,
            transformed_irrelevant=transformed_irrelevant,
            view=transformed_informative + transformed_irrelevant,
        )

    def forward(self, source: torch.Tensor) -> DomainAugmentation:
        embedding = self.encoder(source)
        factor_logits = self.factor_head(embedding)
        return DomainAugmentation(
            source=source,
            embedding=embedding,
            factor_logits=factor_logits,
            view1=self._make_view(source, embedding, factor_logits, 0),
            view2=self._make_view(source, embedding, factor_logits, 1),
        )


def reconstruct_with_phase(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    *,
    sequence_length: int,
) -> torch.Tensor:
    if magnitude.shape != phase.shape:
        raise ValueError("magnitude and phase must have identical shapes")
    if magnitude.is_complex() or phase.is_complex():
        raise TypeError("magnitude and phase must be real tensors")
    spectrum = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    return torch.fft.irfft(spectrum, n=sequence_length, dim=1)


def make_dual_domain_views(
    x: torch.Tensor,
    temporal_sada: SelfAdaptiveDataAugmentation,
    frequency_sada: SelfAdaptiveDataAugmentation,
    config: SaURLConfig,
) -> SaURLViewsAndParts:
    temporal = temporal_sada(x)
    spectrum = torch.fft.rfft(x, n=config.sequence_length, dim=1)
    magnitude = spectrum.abs()
    phase = torch.angle(spectrum)
    frequency = frequency_sada(magnitude)
    frequency_view1 = reconstruct_with_phase(
        frequency.view1.view, phase, sequence_length=config.sequence_length
    )
    frequency_view2 = reconstruct_with_phase(
        frequency.view2.view, phase, sequence_length=config.sequence_length
    )
    return SaURLViewsAndParts(
        temporal=temporal,
        frequency=frequency,
        frequency_phase=phase,
        views=SaURLViews(
            temporal_view1=temporal.view1.view,
            temporal_view2=temporal.view2.view,
            frequency_view1=frequency_view1,
            frequency_view2=frequency_view2,
        ),
    )
