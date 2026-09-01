"""Construction of economically named alpha terminals from task outputs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PrimitiveTable:
    values: dict[str, np.ndarray]

    def __post_init__(self):
        if not self.values:
            raise ValueError("at least one primitive is required")
        lengths = {len(np.asarray(v)) for v in self.values.values()}
        if len(lengths) != 1:
            raise ValueError("all primitives must have the same number of rows")
        for name, value in self.values.items():
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim != 1 or not np.all(np.isfinite(arr)):
                raise ValueError(f"primitive {name!r} must be a finite 1-D array")

    @property
    def n_rows(self) -> int:
        return len(next(iter(self.values.values())))


def _one_dim(name, value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def build_alpha_primitives(*, price_prediction=None, volatility_prediction=None,
                           trend_probabilities=None, trend_logits=None) -> PrimitiveTable:
    """Build aligned terminals from frozen/downstream task predictions.

    Trend columns are expected in bull/neutral/bear order (or an explicitly
    documented project equivalent). Confidence is a margin, not a probability.
    """
    out = {}
    for name, value in (("price_prediction", price_prediction),
                        ("volatility_prediction", volatility_prediction)):
        value = _one_dim(name, value)
        if value is not None:
            out[name] = value
    trend = trend_probabilities if trend_probabilities is not None else trend_logits
    if trend is not None:
        trend = np.asarray(trend, dtype=np.float64)
        if trend.ndim != 2 or trend.shape[1] < 2 or not np.all(np.isfinite(trend)):
            raise ValueError("trend outputs must be a finite [N, classes] array")
        if trend_probabilities is None:
            e = np.exp(trend - trend.max(axis=1, keepdims=True))
            trend = e / e.sum(axis=1, keepdims=True)
        names = ("trend_prob_bull", "trend_prob_neutral", "trend_prob_bear")
        for i in range(min(3, trend.shape[1])):
            out[names[i]] = trend[:, i]
        out["trend_confidence"] = np.sort(trend, axis=1)[:, -1] - np.sort(trend, axis=1)[:, -2]
        if trend.shape[1] >= 3:
            out["trend_direction_margin"] = trend[:, 0] - trend[:, 2]
    return PrimitiveTable(out)
