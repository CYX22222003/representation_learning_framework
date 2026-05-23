from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from features.statistical import batch_statistical_features
from features.transform import batch_transform_features

# Re-exported so callers that previously imported from here continue to work.
from aggregation.aggregator import RepresentationAggregator  # noqa: F401


@dataclass
class FeatureBundle:
    statistical: np.ndarray
    transformed: np.ndarray
    neural: Optional[np.ndarray] = None


def build_feature_bundle(
    sequences: np.ndarray,
    ar_order: int = 5,
    fft_top_k: int = 8,
    wavelet_levels: int = 3,
    neural_embeddings: Optional[np.ndarray] = None,
) -> FeatureBundle:
    statistical = batch_statistical_features(sequences, ar_order=ar_order)
    transformed = batch_transform_features(
        sequences, fft_top_k=fft_top_k, wavelet_levels=wavelet_levels
    )
    if neural_embeddings is not None and len(neural_embeddings) != len(sequences):
        raise ValueError("neural_embeddings must have same first dimension as sequences")
    return FeatureBundle(statistical=statistical, transformed=transformed, neural=neural_embeddings)


class NpzFeatureStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, bundle: FeatureBundle) -> str:
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            self.path,
            statistical=bundle.statistical,
            transformed=bundle.transformed,
            neural=bundle.neural if bundle.neural is not None else np.array([], dtype=np.float32),
        )
        return self.path

    def load(self) -> FeatureBundle:
        data = np.load(self.path)
        neural = data["neural"]
        if neural.size == 0:
            neural = None
        return FeatureBundle(
            statistical=data["statistical"],
            transformed=data["transformed"],
            neural=neural,
        )
