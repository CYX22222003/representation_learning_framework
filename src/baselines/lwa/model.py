"""Two-stage LWA models and the efficient frozen inference extractor.

This module is independently authored from the paper and the project's audited
behavioral specification. No upstream source code is incorporated.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import chain
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LWAConfig
from .encoders import FourierDomainEncoder, TimeDomainEncoder, WaveletDomainEncoder
from .losses import symmetric_nt_xent, two_way_mapping_loss
from .mappers import MapperPair, ProjectionHead


@dataclass(frozen=True)
class DomainTriplet:
    time: torch.Tensor
    fourier: torch.Tensor
    wavelet: torch.Tensor


@dataclass(frozen=True)
class LWAJointLosses:
    time_fourier: torch.Tensor
    time_wavelet: torch.Tensor
    fourier_wavelet: torch.Tensor
    mapped_fourier: torch.Tensor
    mapped_wavelet: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class LWAJointOutput:
    representations: DomainTriplet
    projections: DomainTriplet
    mapped_fourier_projection: torch.Tensor
    mapped_wavelet_projection: torch.Tensor
    losses: LWAJointLosses

    @property
    def total_loss(self) -> torch.Tensor:
        return self.losses.total


class LWAJointModel(nn.Module):
    """Stage A: jointly learn three domains and two projected-space mappings."""

    def __init__(self, config: LWAConfig | None = None) -> None:
        super().__init__()
        self.config = config or LWAConfig()
        self.time_encoder = TimeDomainEncoder(self.config)
        self.fourier_encoder = FourierDomainEncoder(self.config)
        self.wavelet_encoder = WaveletDomainEncoder(self.config)
        self.projectors = nn.ModuleDict(
            {
                name: ProjectionHead(
                    self.config.representation_dim,
                    self.config.projection_hidden_dim,
                    self.config.projection_dim,
                )
                for name in ("time", "fourier", "wavelet")
            }
        )
        self.embedding_mappers = MapperPair(
            self.config.projection_dim,
            self.config.mapper_hidden_channels,
            self.config.mapper_kernel_size,
        )

    @staticmethod
    def _validate_matching_batches(*views: torch.Tensor) -> None:
        if len({view.shape[0] for view in views}) != 1:
            raise ValueError("all LWA domain views must contain the same ordered rows")

    def encode_domains(
        self,
        time_view: torch.Tensor,
        fourier_view: torch.Tensor,
        wavelet_view: torch.Tensor,
    ) -> DomainTriplet:
        self._validate_matching_batches(time_view, fourier_view, wavelet_view)
        return DomainTriplet(
            time=self.time_encoder(time_view),
            fourier=self.fourier_encoder(fourier_view),
            wavelet=self.wavelet_encoder(wavelet_view),
        )

    def project_domains(self, representations: DomainTriplet) -> DomainTriplet:
        return DomainTriplet(
            time=F.normalize(self.projectors["time"](representations.time), dim=1),
            fourier=F.normalize(
                self.projectors["fourier"](representations.fourier), dim=1
            ),
            wavelet=F.normalize(
                self.projectors["wavelet"](representations.wavelet), dim=1
            ),
        )

    def forward(
        self,
        time_view: torch.Tensor,
        fourier_view: torch.Tensor,
        wavelet_view: torch.Tensor,
    ) -> LWAJointOutput:
        representations = self.encode_domains(time_view, fourier_view, wavelet_view)
        projections = self.project_domains(representations)
        mapped_fourier, mapped_wavelet = self.embedding_mappers(projections.time)
        time_fourier = symmetric_nt_xent(
            projections.time,
            projections.fourier,
            temperature=self.config.temperature,
        )
        time_wavelet = symmetric_nt_xent(
            projections.time,
            projections.wavelet,
            temperature=self.config.temperature,
        )
        fourier_wavelet = symmetric_nt_xent(
            projections.fourier,
            projections.wavelet,
            temperature=self.config.temperature,
        )
        map_fourier, map_wavelet, mapping_total = two_way_mapping_loss(
            mapped_fourier,
            projections.fourier,
            mapped_wavelet,
            projections.wavelet,
        )
        total = time_fourier + time_wavelet + fourier_wavelet + mapping_total
        return LWAJointOutput(
            representations=representations,
            projections=projections,
            mapped_fourier_projection=mapped_fourier,
            mapped_wavelet_projection=mapped_wavelet,
            losses=LWAJointLosses(
                time_fourier=time_fourier,
                time_wavelet=time_wavelet,
                fourier_wavelet=fourier_wavelet,
                mapped_fourier=map_fourier,
                mapped_wavelet=map_wavelet,
                total=total,
            ),
        )


@dataclass(frozen=True)
class LWARepresentationMappingOutput:
    representations: DomainTriplet
    mapped_fourier: torch.Tensor
    mapped_wavelet: torch.Tensor
    fourier_loss: torch.Tensor
    wavelet_loss: torch.Tensor
    total_loss: torch.Tensor


class LWARepresentationMapperModel(nn.Module):
    """Stage B: freeze final Stage-A encoders and learn new raw-h mappings."""

    def __init__(self, joint_model: LWAJointModel) -> None:
        super().__init__()
        self.config = joint_model.config
        # A private copy prevents Stage B from mutating the frozen Stage-A state.
        self.time_encoder = copy.deepcopy(joint_model.time_encoder)
        self.fourier_encoder = copy.deepcopy(joint_model.fourier_encoder)
        self.wavelet_encoder = copy.deepcopy(joint_model.wavelet_encoder)
        self.representation_mappers = MapperPair(
            self.config.representation_dim,
            self.config.mapper_hidden_channels,
            self.config.mapper_kernel_size,
        )
        self._freeze_encoders()

    def _freeze_encoders(self) -> None:
        for parameter in chain(
            self.time_encoder.parameters(),
            self.fourier_encoder.parameters(),
            self.wavelet_encoder.parameters(),
        ):
            parameter.requires_grad_(False)
        self.time_encoder.eval()
        self.fourier_encoder.eval()
        self.wavelet_encoder.eval()

    def train(self, mode: bool = True) -> LWARepresentationMapperModel:
        super().train(mode)
        self._freeze_encoders()
        return self

    def mapper_parameters(self) -> Iterable[nn.Parameter]:
        return self.representation_mappers.parameters()

    def forward(
        self,
        time_view: torch.Tensor,
        fourier_view: torch.Tensor,
        wavelet_view: torch.Tensor,
    ) -> LWARepresentationMappingOutput:
        LWAJointModel._validate_matching_batches(time_view, fourier_view, wavelet_view)
        with torch.no_grad():
            representations = DomainTriplet(
                time=self.time_encoder(time_view),
                fourier=self.fourier_encoder(fourier_view),
                wavelet=self.wavelet_encoder(wavelet_view),
            )
        mapped_fourier, mapped_wavelet = self.representation_mappers(
            representations.time
        )
        fourier_loss, wavelet_loss, total = two_way_mapping_loss(
            mapped_fourier,
            representations.fourier,
            mapped_wavelet,
            representations.wavelet,
        )
        return LWARepresentationMappingOutput(
            representations=representations,
            mapped_fourier=mapped_fourier,
            mapped_wavelet=mapped_wavelet,
            fourier_loss=fourier_loss,
            wavelet_loss=wavelet_loss,
            total_loss=total,
        )


@dataclass(frozen=True)
class LWAInferenceParts:
    time: torch.Tensor
    mapped_fourier: torch.Tensor
    mapped_wavelet: torch.Tensor
    concatenated: torch.Tensor


class LWAInferenceEncoder(nn.Module):
    """Efficient frozen extractor containing only the paper's retained modules."""

    def __init__(self, mapper_model: LWARepresentationMapperModel) -> None:
        super().__init__()
        self.config = mapper_model.config
        self.time_encoder = copy.deepcopy(mapper_model.time_encoder)
        self.representation_mappers = copy.deepcopy(
            mapper_model.representation_mappers
        )
        self.output_dim = self.config.inference_dim
        self._freeze()

    def _freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> LWAInferenceEncoder:
        # The extractor is permanently frozen and BatchNorm/dropout stay in eval mode.
        self._freeze()
        return self

    @torch.no_grad()
    def encode_parts(self, time_view: torch.Tensor) -> LWAInferenceParts:
        time = self.time_encoder(time_view)
        mapped_fourier, mapped_wavelet = self.representation_mappers(time)
        concatenated = torch.cat((time, mapped_fourier, mapped_wavelet), dim=1)
        return LWAInferenceParts(
            time=time,
            mapped_fourier=mapped_fourier,
            mapped_wavelet=mapped_wavelet,
            concatenated=concatenated,
        )

    def forward(self, time_view: torch.Tensor) -> torch.Tensor:
        return self.encode_parts(time_view).concatenated


def build_lwa_joint(config: LWAConfig | None = None) -> LWAJointModel:
    return LWAJointModel(config)


def build_lwa_mapper_stage(joint_model: LWAJointModel) -> LWARepresentationMapperModel:
    return LWARepresentationMapperModel(joint_model)


def build_lwa_inference_encoder(
    mapper_model: LWARepresentationMapperModel,
) -> LWAInferenceEncoder:
    return LWAInferenceEncoder(mapper_model)
