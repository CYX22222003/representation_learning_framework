"""Phase 6.9 source-aligned xLSTM-Mixer complete-system baseline."""

from .config import (
    DEFAULT_XLSTM_MIXER_CONFIG,
    METHOD_ID,
    SOURCE_CONTRACT,
    XLSTMMixerConfig,
    XLSTMMixerSourceContract,
)
from .model import XLSTMMixer, XLSTMMixerTrace
from .normalization import NonAffineRevIN, RevINStatistics

__all__ = [
    "DEFAULT_XLSTM_MIXER_CONFIG",
    "METHOD_ID",
    "NonAffineRevIN",
    "RevINStatistics",
    "SOURCE_CONTRACT",
    "XLSTMMixer",
    "XLSTMMixerConfig",
    "XLSTMMixerSourceContract",
    "XLSTMMixerTrace",
]
