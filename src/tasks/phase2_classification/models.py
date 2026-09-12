from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from aggregation.aggregator import RepresentationAggregator
from baselines.mlp_baseline.mlp_model import RawOHLCVMLP
from baselines.ta_mlp_baseline.ta_mlp_model import TAMLPClassifier
from tasks.trend_classification import TrendClassifier


class FrameworkMovementClassifier(nn.Module):
    def __init__(
        self,
        branch_dims: Mapping[str, int],
        *,
        mode: str = "concat",
        out_dim: int = 128,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.branch_dims = dict(branch_dims)
        self.aggregator = RepresentationAggregator(self.branch_dims, out_dim=out_dim, mode=mode)
        self.head = TrendClassifier(self.aggregator.output_dim, hidden_dim=head_hidden_dim, n_classes=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = 0
        branches = {}
        for name, width in self.branch_dims.items():
            branches[name] = x[:, offset : offset + width]
            offset += width
        embedding, _ = self.aggregator(branches)
        return self.head(embedding)


class RawMovementMLP(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        *,
        hidden_dims: list[int] | None = None,
        encoder_output_dim: int = 128,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = RawOHLCVMLP(
            seq_len=seq_len, n_features=n_features, hidden_dims=hidden_dims,
            output_dim=encoder_output_dim, dropout=dropout,
        )
        self.head = TrendClassifier(self.encoder.output_dim, hidden_dim=head_hidden_dim, n_classes=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def make_ta_mlp(input_dim: int) -> TAMLPClassifier:
    return TAMLPClassifier(in_dim=input_dim, n_classes=3)
