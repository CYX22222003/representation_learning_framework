"""LWA-Frozen paper-guided independent implementation for Phase 6.7."""

from .blocks import ConvNormReLU2d, ResidualBlock1D, SamePadConv1d, SamePadMaxPool1d
from .config import DEFAULT_LWA_CONFIG, LWAConfig
from .encoders import FourierDomainEncoder, TimeDomainEncoder, WaveletDomainEncoder
from .losses import (
    mean_sample_l1,
    negative_pair_mask,
    positive_pair_indices,
    symmetric_nt_xent,
    two_way_mapping_loss,
)
from .mappers import ConvolutionalMapper, MapperPair, ProjectionHead
from .model import (
    DomainTriplet,
    LWAInferenceEncoder,
    LWAInferenceParts,
    LWAJointLosses,
    LWAJointModel,
    LWAJointOutput,
    LWARepresentationMapperModel,
    LWARepresentationMappingOutput,
    build_lwa_inference_encoder,
    build_lwa_joint,
    build_lwa_mapper_stage,
)
from .transforms import MorletCWT, orthonormal_rfft, validate_time_batch

__all__ = [
    "DEFAULT_LWA_CONFIG",
    "ConvNormReLU2d",
    "ConvolutionalMapper",
    "DomainTriplet",
    "FourierDomainEncoder",
    "LWAConfig",
    "LWAInferenceEncoder",
    "LWAInferenceParts",
    "LWAJointLosses",
    "LWAJointModel",
    "LWAJointOutput",
    "LWARepresentationMapperModel",
    "LWARepresentationMappingOutput",
    "MapperPair",
    "MorletCWT",
    "ProjectionHead",
    "ResidualBlock1D",
    "SamePadConv1d",
    "SamePadMaxPool1d",
    "TimeDomainEncoder",
    "WaveletDomainEncoder",
    "build_lwa_inference_encoder",
    "build_lwa_joint",
    "build_lwa_mapper_stage",
    "mean_sample_l1",
    "negative_pair_mask",
    "orthonormal_rfft",
    "positive_pair_indices",
    "symmetric_nt_xent",
    "two_way_mapping_loss",
    "validate_time_batch",
]
