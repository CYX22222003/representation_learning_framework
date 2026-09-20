"""External raw-data acquisition utilities.

This package is deliberately separate from ``src.data_processing``.  It
downloads and records source data; it does not establish experiment splits,
fit preprocessing, or construct model samples.
"""

from .findata import (
    FinDataClient,
    aggregate_one_hour_to_four_hour,
    build_market_candidates,
    load_lumid_token,
)
from .polymarket_pipeline import (
    CANDLE_QUARANTINE_RULE_VERSION,
    CandleQuarantineResult,
    quarantine_condition_candles,
)

__all__ = [
    "FinDataClient",
    "aggregate_one_hour_to_four_hour",
    "build_market_candidates",
    "CANDLE_QUARANTINE_RULE_VERSION",
    "CandleQuarantineResult",
    "load_lumid_token",
    "quarantine_condition_candles",
]
