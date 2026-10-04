from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import chain
from typing import Iterable

import torch
import torch.nn as nn

from .attention import AttentionOutput, RepresentationWiseAttention
from .augmentation import (
    SaURLViews,
    SaURLViewsAndParts,
    SelfAdaptiveDataAugmentation,
    make_dual_domain_views,
)
from .config import SaURLConfig
from .encoder import DilatedCNNEncoder
from .losses import SaDALosses, sada_objective, symmetric_byol_loss


BRANCH_NAMES: tuple[str, ...] = ("time", "frequency", "cross")


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class BundleRepresentations:
    branches: torch.Tensor
    attention_weights: torch.Tensor
    weighted_branches: torch.Tensor
    fused: torch.Tensor


@dataclass(frozen=True)
class SaSSLLosses:
    time: torch.Tensor
    frequency: torch.Tensor
    cross: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class SaSSLOutputs:
    online_a: BundleRepresentations
    online_b: BundleRepresentations
    target_a: BundleRepresentations
    target_b: BundleRepresentations
    online_predictions_a: dict[str, torch.Tensor]
    online_predictions_b: dict[str, torch.Tensor]
    target_projections_a: dict[str, torch.Tensor]
    target_projections_b: dict[str, torch.Tensor]
    losses: SaSSLLosses

    @property
    def total_loss(self) -> torch.Tensor:
        return self.losses.total


class SaURLModel(nn.Module):
    """Independent implementation of the frozen Phase 6.7 SaURL contract."""

    def __init__(self, config: SaURLConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_sada = SelfAdaptiveDataAugmentation(config)
        self.frequency_sada = SelfAdaptiveDataAugmentation(config)

        def make_encoder() -> DilatedCNNEncoder:
            return DilatedCNNEncoder(
                input_dim=config.input_dim,
                hidden_dim=config.encoder_hidden_dim,
                output_dim=config.representation_dim,
                dilations=config.dilations,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            )

        self.online_encoders = nn.ModuleDict(
            {branch: make_encoder() for branch in BRANCH_NAMES}
        )
        self.online_projectors = nn.ModuleDict(
            {
                branch: _make_mlp(
                    config.representation_dim,
                    config.projector_hidden_dim,
                    config.projection_dim,
                )
                for branch in BRANCH_NAMES
            }
        )
        self.online_predictors = nn.ModuleDict(
            {
                branch: _make_mlp(
                    config.projection_dim,
                    config.predictor_hidden_dim,
                    config.projection_dim,
                )
                for branch in BRANCH_NAMES
            }
        )
        self.target_encoders = copy.deepcopy(self.online_encoders)
        self.target_projectors = copy.deepcopy(self.online_projectors)
        self.attention = RepresentationWiseAttention(
            representation_dim=config.representation_dim,
            regions=config.attention_regions,
        )
        self._freeze_targets()

    def _freeze_targets(self) -> None:
        for parameter in chain(
            self.target_encoders.parameters(), self.target_projectors.parameters()
        ):
            parameter.requires_grad_(False)
        self.target_encoders.eval()
        self.target_projectors.eval()

    def train(self, mode: bool = True) -> SaURLModel:
        super().train(mode)
        # Target dropout must remain disabled and target parameters stay frozen.
        self._freeze_targets()
        return self

    def _validate_batch(self, batch: torch.Tensor) -> None:
        expected_tail = (self.config.sequence_length, self.config.input_dim)
        if batch.ndim != 3 or tuple(batch.shape[1:]) != expected_tail:
            raise ValueError(
                f"expected [batch, {self.config.sequence_length}, {self.config.input_dim}], "
                f"got {tuple(batch.shape)}"
            )
        if not batch.is_floating_point() or batch.is_complex():
            raise TypeError("SaURL inputs must be real floating-point tensors")

    def make_views(self, batch: torch.Tensor) -> SaURLViewsAndParts:
        self._validate_batch(batch)
        return make_dual_domain_views(
            batch,
            self.temporal_sada,
            self.frequency_sada,
            self.config,
        )

    def sada_losses(self, views_and_parts: SaURLViewsAndParts) -> SaDALosses:
        return sada_objective(
            views_and_parts.temporal,
            views_and_parts.frequency,
            self.config,
        )

    @staticmethod
    def _bundle_inputs(
        views: SaURLViews,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        bundle_a = {
            "time": views.temporal_view1,
            "frequency": views.frequency_view1,
            "cross": views.temporal_view1,
        }
        bundle_b = {
            "time": views.temporal_view2,
            "frequency": views.frequency_view2,
            "cross": views.frequency_view1,
        }
        return bundle_a, bundle_b

    @staticmethod
    def _as_bundle(
        branch_vectors: dict[str, torch.Tensor], attention: AttentionOutput
    ) -> BundleRepresentations:
        return BundleRepresentations(
            branches=torch.stack([branch_vectors[name] for name in BRANCH_NAMES], dim=1),
            attention_weights=attention.weights,
            weighted_branches=attention.weighted_branches,
            fused=attention.fused,
        )

    def _online_bundle(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[BundleRepresentations, dict[str, torch.Tensor]]:
        branch_vectors = {
            branch: self.online_encoders[branch](inputs[branch])
            for branch in BRANCH_NAMES
        }
        stacked = torch.stack([branch_vectors[name] for name in BRANCH_NAMES], dim=1)
        attended = self.attention(stacked)
        predictions = {
            branch: self.online_predictors[branch](
                self.online_projectors[branch](attended.weighted_branches[:, index])
            )
            for index, branch in enumerate(BRANCH_NAMES)
        }
        return self._as_bundle(branch_vectors, attended), predictions

    @torch.no_grad()
    def _target_bundle(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[BundleRepresentations, dict[str, torch.Tensor]]:
        branch_vectors = {
            branch: self.target_encoders[branch](inputs[branch])
            for branch in BRANCH_NAMES
        }
        stacked = torch.stack([branch_vectors[name] for name in BRANCH_NAMES], dim=1)
        attended = self.attention(stacked)
        projections = {
            branch: self.target_projectors[branch](
                attended.weighted_branches[:, index]
            )
            for index, branch in enumerate(BRANCH_NAMES)
        }
        return self._as_bundle(branch_vectors, attended), projections

    def sassl_forward(self, views: SaURLViews) -> SaSSLOutputs:
        for view in (
            views.temporal_view1,
            views.temporal_view2,
            views.frequency_view1,
            views.frequency_view2,
        ):
            self._validate_batch(view)
        batch_sizes = {
            views.temporal_view1.shape[0],
            views.temporal_view2.shape[0],
            views.frequency_view1.shape[0],
            views.frequency_view2.shape[0],
        }
        if len(batch_sizes) != 1:
            raise ValueError("all SaSSL views must contain the same ordered rows")
        bundle_a_inputs, bundle_b_inputs = self._bundle_inputs(views)
        online_a, predictions_a = self._online_bundle(bundle_a_inputs)
        online_b, predictions_b = self._online_bundle(bundle_b_inputs)
        target_a, target_projections_a = self._target_bundle(bundle_a_inputs)
        target_b, target_projections_b = self._target_bundle(bundle_b_inputs)

        branch_losses = {
            branch: symmetric_byol_loss(
                predictions_a[branch],
                target_projections_b[branch],
                predictions_b[branch],
                target_projections_a[branch],
            )
            for branch in BRANCH_NAMES
        }
        losses = SaSSLLosses(
            time=branch_losses["time"],
            frequency=branch_losses["frequency"],
            cross=branch_losses["cross"],
            total=sum(branch_losses.values()),
        )
        return SaSSLOutputs(
            online_a=online_a,
            online_b=online_b,
            target_a=target_a,
            target_b=target_b,
            online_predictions_a=predictions_a,
            online_predictions_b=predictions_b,
            target_projections_a=target_projections_a,
            target_projections_b=target_projections_b,
            losses=losses,
        )

    def encode_parts(self, batch: torch.Tensor) -> BundleRepresentations:
        """Return online pre-projector inference parts for diagnostics/replay."""

        self._validate_batch(batch)
        branch_vectors = {
            branch: self.online_encoders[branch](batch) for branch in BRANCH_NAMES
        }
        stacked = torch.stack([branch_vectors[name] for name in BRANCH_NAMES], dim=1)
        return self._as_bundle(branch_vectors, self.attention(stacked))

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Return the 128-wide RwAM sum; SaDA and BYOL heads are excluded."""

        return self.encode_parts(batch).fused

    @torch.no_grad()
    def update_targets(self, tau: float | None = None) -> None:
        coefficient = self.config.ema_tau if tau is None else tau
        if not 0.0 <= coefficient <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        pairs = (
            (self.online_encoders, self.target_encoders),
            (self.online_projectors, self.target_projectors),
        )
        for online_modules, target_modules in pairs:
            for online_parameter, target_parameter in zip(
                online_modules.parameters(), target_modules.parameters()
            ):
                target_parameter.mul_(coefficient).add_(
                    online_parameter, alpha=1.0 - coefficient
                )

    def sada_parameters(self) -> Iterable[nn.Parameter]:
        return chain(self.temporal_sada.parameters(), self.frequency_sada.parameters())

    def sassl_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.online_encoders.parameters(),
            self.online_projectors.parameters(),
            self.online_predictors.parameters(),
            self.attention.parameters(),
        )


def build_saurl(config: SaURLConfig | None = None) -> SaURLModel:
    return SaURLModel(config or SaURLConfig())
