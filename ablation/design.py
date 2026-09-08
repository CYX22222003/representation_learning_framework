"""Pure experiment-matrix definitions for the independent ablation study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


CANONICAL_BRANCHES = ("statistical", "transformed", "vae", "contrastive", "byol")
CANONICAL_TASKS = ("price_prediction", "volatility_prediction", "trend_classification")


@dataclass(frozen=True)
class AblationVariant:
    """One predeclared branch configuration in the ablation matrix."""

    name: str
    branches: tuple[str, ...]
    mode: str
    family: str
    estimand: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["branches"] = list(self.branches)
        return payload


def validate_available_branches(
    available: Mapping[str, int] | Iterable[str],
    required: Iterable[str] = CANONICAL_BRANCHES,
) -> None:
    names = set(available)
    missing = sorted(set(required).difference(names))
    if missing:
        raise ValueError(f"Feature store is missing required ablation branches: {missing}")


def build_variants(
    branches: tuple[str, ...] = CANONICAL_BRANCHES,
    families: tuple[str, ...] = ("single", "leave_one_out", "full"),
    include_gated: bool = False,
) -> list[AblationVariant]:
    """Build a deterministic, uniquely named experiment matrix.

    Single-branch probes estimate standalone utility. Leave-one-out probes
    estimate a branch's conditional contribution relative to the full concat
    control. Gated fusion is optional because it changes trainable capacity.
    """
    if not branches or len(set(branches)) != len(branches):
        raise ValueError("branches must be non-empty and unique")
    allowed = {"single", "leave_one_out", "full"}
    unknown = set(families).difference(allowed)
    if unknown or not families:
        raise ValueError(f"families must be a non-empty subset of {sorted(allowed)}")
    if include_gated and "full" not in families:
        raise ValueError("include_gated requires the full family and its matched concat control")

    variants: list[AblationVariant] = []
    if "single" in families:
        variants.extend(
            AblationVariant(
                name=f"single_{branch}",
                branches=(branch,),
                mode="concat",
                family="single",
                estimand=f"standalone utility of {branch}",
            )
            for branch in branches
        )
    if "leave_one_out" in families:
        variants.extend(
            AblationVariant(
                name=f"without_{branch}",
                branches=tuple(item for item in branches if item != branch),
                mode="concat",
                family="leave_one_out",
                estimand=f"conditional contribution of {branch} given the other branches",
            )
            for branch in branches
        )
    if "full" in families:
        variants.append(
            AblationVariant(
                name="full_concat",
                branches=branches,
                mode="concat",
                family="control",
                estimand="matched full-representation control",
            )
        )
        if include_gated:
            variants.append(
                AblationVariant(
                    name="full_gated",
                    branches=branches,
                    mode="gated",
                    family="fusion",
                    estimand="fusion comparison with additional trainable capacity",
                )
            )
    names = [variant.name for variant in variants]
    if len(names) != len(set(names)):
        raise AssertionError("Ablation variant names must be unique")
    return variants
