from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from features.statistical import batch_statistical_features
from features.transform import batch_transform_features

# Re-exported so callers that previously imported from here continue to work.
from aggregation.aggregator import RepresentationAggregator  # noqa: F401


@dataclass(init=False)
class FeatureBundle:
    """
    Container for all branch features belonging to a set of sequences.

    statistical     : [N, stat_dim]            — AR + GARCH features (deterministic)
    transformed     : [N, trans_dim]           — FFT + Haar wavelet features (deterministic)
    neural_branches : dict[str, [N, dim_i]]    — named frozen neural embeddings

    Neural encoders are stored as separate named branches (for example
    ``"vae"`` and ``"contrastive"``) so the aggregator and ablation code can
    preserve branch identity. The ``neural`` argument/property remains as a
    compatibility shim for older callers that packed every neural embedding into
    one matrix.
    """
    statistical: np.ndarray
    transformed: np.ndarray
    neural_branches: dict[str, np.ndarray] = field(default_factory=dict)

    def __init__(
        self,
        statistical: np.ndarray,
        transformed: np.ndarray,
        neural_branches: Optional[Mapping[str, np.ndarray] | np.ndarray] = None,
        neural: Optional[np.ndarray] = None,
    ) -> None:
        if (
            neural is None
            and neural_branches is not None
            and not isinstance(neural_branches, Mapping)
        ):
            neural = neural_branches
            neural_branches = None
        if neural is not None and neural_branches:
            raise ValueError("Pass either neural_branches or legacy neural, not both")
        self.statistical = np.asarray(statistical, dtype=np.float32)
        self.transformed = np.asarray(transformed, dtype=np.float32)

        branches: Mapping[str, np.ndarray]
        if neural_branches is not None:
            branches = neural_branches
        elif neural is not None:
            branches = {"neural": neural}
        else:
            branches = {}
        self.neural_branches = {
            str(name): np.asarray(values, dtype=np.float32)
            for name, values in branches.items()
        }
        self._validate()

    def _validate(self) -> None:
        if self.statistical.ndim != 2:
            raise ValueError("statistical features must be 2D [N, dim]")
        if self.transformed.ndim != 2:
            raise ValueError("transformed features must be 2D [N, dim]")
        if len(self.statistical) != len(self.transformed):
            raise ValueError("statistical and transformed features must have the same row count")

        reserved = {"statistical", "transformed"}
        for name, values in self.neural_branches.items():
            if not name:
                raise ValueError("neural branch names must be non-empty")
            if name in reserved:
                raise ValueError(f"{name!r} is reserved for deterministic features")
            if values.ndim != 2:
                raise ValueError(f"neural branch {name!r} must be 2D [N, dim]")
            if len(values) != len(self.statistical):
                raise ValueError(
                    f"neural branch {name!r} must have the same row count as deterministic features"
                )

    @property
    def neural(self) -> Optional[np.ndarray]:
        """Legacy packed neural matrix, or ``None`` when no neural branches exist."""
        if not self.neural_branches:
            return None
        return np.concatenate(list(self.neural_branches.values()), axis=1)

    def as_branch_dict(self) -> dict[str, np.ndarray]:
        """Return a branch dictionary suitable for ``RepresentationAggregator``."""
        return {
            "statistical": self.statistical,
            "transformed": self.transformed,
            **self.neural_branches,
        }


def build_feature_bundle(
    sequences: np.ndarray,
    ar_order: int = 5,
    fft_top_k: int = 8,
    wavelet_levels: int = 3,
    neural_embeddings: Optional[np.ndarray] = None,
    neural_branches: Optional[Mapping[str, np.ndarray]] = None,
) -> FeatureBundle:
    statistical = batch_statistical_features(sequences, ar_order=ar_order)
    transformed = batch_transform_features(
        sequences, fft_top_k=fft_top_k, wavelet_levels=wavelet_levels
    )
    if neural_embeddings is not None and neural_branches:
        raise ValueError("Pass either neural_embeddings or neural_branches, not both")
    if neural_embeddings is not None and len(neural_embeddings) != len(sequences):
        raise ValueError("neural_embeddings must have same first dimension as sequences")
    if neural_branches is not None:
        for name, values in neural_branches.items():
            if len(values) != len(sequences):
                raise ValueError(
                    f"neural branch {name!r} must have same first dimension as sequences"
                )
    return FeatureBundle(
        statistical=statistical,
        transformed=transformed,
        neural=neural_embeddings,
        neural_branches=neural_branches,
    )


class NpzFeatureStore:
    """
    Saves and loads a FeatureBundle to/from a compressed .npz file.

    The deterministic branches are always saved as ``statistical`` and
    ``transformed``. Frozen neural embeddings are saved as separate named arrays
    such as ``vae`` and ``contrastive``. Legacy files with a packed ``neural``
    array are still readable; an empty legacy ``neural`` array is ignored.
    """

    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, bundle: FeatureBundle) -> str:
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "statistical": bundle.statistical,
            "transformed": bundle.transformed,
            **bundle.neural_branches,
        }
        np.savez_compressed(self.path, **payload)
        return self.path

    def load(self) -> FeatureBundle:
        with np.load(self.path) as data:
            reserved = {"statistical", "transformed"}
            if "statistical" not in data or "transformed" not in data:
                raise ValueError("Feature store must contain 'statistical' and 'transformed'")
            neural_branches = {}
            for key in data.files:
                if key in reserved:
                    continue
                values = data[key]
                if key == "neural" and values.size == 0:
                    continue
                neural_branches[key] = values
            return FeatureBundle(
                statistical=data["statistical"],
                transformed=data["transformed"],
                neural_branches=neural_branches,
            )
