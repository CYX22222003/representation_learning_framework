"""Leakage-aware skeletons for future alpha-factor research.

Alpha mining is deliberately downstream of the task heads: formulas consume
economically named predictions, never arbitrary representation dimensions.
"""

from .primitives import PrimitiveTable, build_alpha_primitives
from .oof import OOFResult, expanding_folds, generate_oof_predictions
from .formula import Expression, candidate_formulas
from .scoring import FactorScore, score_factor
from .selection import select_non_redundant
from .pipeline import MiningConfig, mine_from_oof

__all__ = [
    "PrimitiveTable", "build_alpha_primitives", "OOFResult", "expanding_folds",
    "generate_oof_predictions", "Expression", "candidate_formulas",
    "FactorScore", "score_factor", "select_non_redundant",
    "MiningConfig", "mine_from_oof",
]
"""Training-only alpha-factor research utilities."""
