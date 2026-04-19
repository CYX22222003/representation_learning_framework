from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from features.statistical import batch_statistical_features
from features.transform import batch_transform_features


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


class RepresentationAggregator(nn.Module):
    """
    Adaptive weighted aggregation over statistical, transformed, and neural branches.
    """

    def __init__(
        self,
        statistical_dim: int,
        transformed_dim: int,
        neural_dim: int,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.stat_branch = nn.Sequential(nn.Linear(statistical_dim, out_dim), nn.GELU())
        self.trans_branch = nn.Sequential(nn.Linear(transformed_dim, out_dim), nn.GELU())
        self.neural_branch = nn.Sequential(nn.Linear(neural_dim, out_dim), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(out_dim * 3, out_dim), nn.GELU(), nn.Linear(out_dim, 3))

    def forward(
        self,
        statistical_repr: torch.Tensor,
        transformed_repr: torch.Tensor,
        neural_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.stat_branch(statistical_repr)
        t = self.trans_branch(transformed_repr)
        n = self.neural_branch(neural_repr)

        combined = torch.cat([s, t, n], dim=-1)
        weights = torch.softmax(self.gate(combined), dim=-1)
        aggregated = (
            weights[:, 0:1] * s + weights[:, 1:2] * t + weights[:, 2:3] * n
        )
        return aggregated, weights


class NpzFeatureStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, bundle: FeatureBundle) -> str:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
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
