from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .config import SaURLConfig

if TYPE_CHECKING:
    from .augmentation import DomainAugmentation, ViewParts


MMD_BANDWIDTH_SCALES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def _flatten_samples(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("MMD inputs must include batch and feature dimensions")
    if x.shape[0] < 2:
        raise ValueError("MMD requires at least two samples per input")
    if not x.is_floating_point():
        raise TypeError("MMD inputs must be floating-point tensors")
    return x.reshape(x.shape[0], -1)


def _pairwise_squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    distances = (
        x.square().sum(dim=1, keepdim=True)
        + y.square().sum(dim=1).unsqueeze(0)
        - 2.0 * x @ y.transpose(0, 1)
    )
    return distances.clamp_min(0.0)


def five_kernel_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Biased five-kernel Gaussian MMD with a detached robust bandwidth."""

    x_flat = _flatten_samples(x)
    y_flat = _flatten_samples(y)
    if x_flat.shape[1] != y_flat.shape[1]:
        raise ValueError("MMD inputs must have the same flattened feature width")
    combined = torch.cat((x_flat, y_flat), dim=0)
    combined_distances = _pairwise_squared_distance(combined, combined)
    count = combined.shape[0]
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=combined.device)
    bandwidth = combined_distances[off_diagonal].mean().detach().clamp_min(1e-8)

    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        squared_distance = _pairwise_squared_distance(left, right)
        values = torch.zeros_like(squared_distance)
        for scale in MMD_BANDWIDTH_SCALES:
            values = values + torch.exp(-squared_distance / (2.0 * bandwidth * scale))
        return values

    return (
        kernel(x_flat, x_flat).mean()
        + kernel(y_flat, y_flat).mean()
        - kernel(x_flat, y_flat).mean()
        - kernel(y_flat, x_flat).mean()
    )


def temporal_total_variation(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 3 or mask.shape[-1] != 1:
        raise ValueError("mask must have shape [batch, positions, 1]")
    if mask.shape[1] < 2:
        return mask.new_zeros(())
    return (mask[:, 1:] - mask[:, :-1]).abs().mean()


def byol_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (prediction * target).sum(dim=-1).mean()


def symmetric_byol_loss(
    prediction_a: torch.Tensor,
    target_b: torch.Tensor,
    prediction_b: torch.Tensor,
    target_a: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        byol_prediction_loss(prediction_a, target_b)
        + byol_prediction_loss(prediction_b, target_a)
    )


@dataclass(frozen=True)
class ViewSaDALoss:
    mask_cardinality: torch.Tensor
    preservation: torch.Tensor
    continuity: torch.Tensor
    dissimilarity: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class SaDALosses:
    temporal_view1: ViewSaDALoss
    temporal_view2: ViewSaDALoss
    frequency_view1: ViewSaDALoss
    frequency_view2: ViewSaDALoss
    temporal_diversity: torch.Tensor
    frequency_diversity: torch.Tensor
    total: torch.Tensor


def _view_sada_loss(
    source: torch.Tensor,
    parts: ViewParts,
    *,
    include_continuity: bool,
    config: SaURLConfig,
) -> ViewSaDALoss:
    cardinality = parts.mask.mean()
    preservation = five_kernel_mmd(source, parts.transformed_informative)
    continuity = (
        temporal_total_variation(parts.mask)
        if include_continuity
        else source.new_zeros(())
    )
    dissimilarity = -five_kernel_mmd(parts.irrelevant, parts.transformed_irrelevant)
    total = (
        cardinality
        + config.sada_alpha * preservation
        + config.sada_beta * continuity
        + config.sada_gamma * dissimilarity
    )
    return ViewSaDALoss(
        mask_cardinality=cardinality,
        preservation=preservation,
        continuity=continuity,
        dissimilarity=dissimilarity,
        total=total,
    )


def sada_objective(
    temporal: DomainAugmentation,
    frequency: DomainAugmentation,
    config: SaURLConfig,
) -> SaDALosses:
    temporal_view1 = _view_sada_loss(
        temporal.source, temporal.view1, include_continuity=True, config=config
    )
    temporal_view2 = _view_sada_loss(
        temporal.source, temporal.view2, include_continuity=True, config=config
    )
    frequency_view1 = _view_sada_loss(
        frequency.source, frequency.view1, include_continuity=False, config=config
    )
    frequency_view2 = _view_sada_loss(
        frequency.source, frequency.view2, include_continuity=False, config=config
    )
    temporal_diversity = -five_kernel_mmd(temporal.view1.view, temporal.view2.view)
    frequency_diversity = -five_kernel_mmd(frequency.view1.view, frequency.view2.view)
    augmentation = 0.5 * (
        temporal_view1.total
        + temporal_view2.total
        + frequency_view1.total
        + frequency_view2.total
    )
    total = augmentation + config.diversity_weight * (
        temporal_diversity + frequency_diversity
    )
    return SaDALosses(
        temporal_view1=temporal_view1,
        temporal_view2=temporal_view2,
        frequency_view1=frequency_view1,
        frequency_view2=frequency_view2,
        temporal_diversity=temporal_diversity,
        frequency_diversity=frequency_diversity,
        total=total,
    )
