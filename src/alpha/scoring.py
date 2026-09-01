"""Training-only factor diagnostics (not a trading backtest)."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FactorScore:
    ic: float
    rank_ic: float
    n_obs: int
    block_ic: tuple[float, ...]


def _corr(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def score_factor(factor, target, *, n_blocks=5):
    x, y = np.asarray(factor, float).reshape(-1), np.asarray(target, float).reshape(-1)
    if len(x) != len(y): raise ValueError("factor and target must be aligned")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0: raise ValueError("no finite observations")
    rx = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    ry = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
    edges = np.linspace(0, len(x), min(n_blocks, len(x)) + 1, dtype=int)
    blocks = tuple(_corr(x[a:b], y[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a)
    return FactorScore(_corr(x, y), _corr(rx, ry), len(x), blocks)
