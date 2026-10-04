"""SaURL-TS-Frozen paper-guided reimplementation for Phase 6.7."""

from .adapter import SaURLAdapter, SaURLInputScaler, build_saurl_adapter
from .attention import AttentionOutput, RepresentationWiseAttention
from .augmentation import (
    DomainAugmentation,
    SaURLViews,
    SaURLViewsAndParts,
    SelfAdaptiveDataAugmentation,
    ViewParts,
    reconstruct_with_phase,
)
from .config import DEFAULT_SAURL_CONFIG, SaURLConfig
from .encoder import DilatedCNNEncoder, DilatedResidualBlock
from .losses import (
    SaDALosses,
    ViewSaDALoss,
    byol_prediction_loss,
    five_kernel_mmd,
    symmetric_byol_loss,
    temporal_total_variation,
)
from .model import (
    BRANCH_NAMES,
    BundleRepresentations,
    SaSSLLosses,
    SaSSLOutputs,
    SaURLModel,
    build_saurl,
)

__all__ = [
    "AttentionOutput",
    "BRANCH_NAMES",
    "BundleRepresentations",
    "DEFAULT_SAURL_CONFIG",
    "DilatedCNNEncoder",
    "DilatedResidualBlock",
    "DomainAugmentation",
    "RepresentationWiseAttention",
    "SaDALosses",
    "SaSSLLosses",
    "SaSSLOutputs",
    "SaURLAdapter",
    "SaURLConfig",
    "SaURLInputScaler",
    "SaURLModel",
    "SaURLViews",
    "SaURLViewsAndParts",
    "SelfAdaptiveDataAugmentation",
    "ViewParts",
    "ViewSaDALoss",
    "build_saurl",
    "build_saurl_adapter",
    "byol_prediction_loss",
    "five_kernel_mmd",
    "reconstruct_with_phase",
    "symmetric_byol_loss",
    "temporal_total_variation",
]
