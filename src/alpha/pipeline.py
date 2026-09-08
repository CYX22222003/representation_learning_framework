"""Training-only orchestration for a future alpha-factor mining run."""
from dataclasses import dataclass
import numpy as np
from .formula import candidate_formulas
from .scoring import score_factor
from .selection import select_non_redundant


@dataclass(frozen=True)
class MiningConfig:
    max_depth: int = 2
    max_factors: int = 10
    correlation_threshold: float = 0.9
    n_blocks: int = 5


def mine_from_oof(primitives, target, config=MiningConfig()):
    """Enumerate and rank formulas using OOF rows only.

    ``primitives`` should be a ``PrimitiveTable`` or a mapping of arrays. The
    caller remains responsible for producing OOF predictions and for holding
    back a fresh evaluation period.
    """
    values = primitives.values if hasattr(primitives, "values") else primitives
    formulas = candidate_formulas(values.keys(), max_depth=config.max_depth)
    candidates, scores = {}, {}
    for expr in formulas:
        name = expr.to_string()
        if name in candidates:
            continue
        value = expr.evaluate(values)
        if not np.all(np.isfinite(value)) or np.std(value) == 0:
            continue
        score = score_factor(value, target, n_blocks=config.n_blocks)
        candidates[name], scores[name] = value, score
    selected = select_non_redundant(candidates, scores,
                                    max_factors=config.max_factors,
                                    correlation_threshold=config.correlation_threshold)
    return {"selected": selected, "candidates": candidates, "scores": scores}
