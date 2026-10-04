from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AttentionOutput:
    weights: torch.Tensor
    regional_weights: torch.Tensor
    weighted_branches: torch.Tensor
    fused: torch.Tensor


class RepresentationWiseAttention(nn.Module):
    """Eight-region representation-wise attention reconstruction."""

    def __init__(self, representation_dim: int = 128, regions: int = 8) -> None:
        super().__init__()
        if representation_dim <= 0 or regions <= 0:
            raise ValueError("representation_dim and regions must be positive")
        if representation_dim % regions != 0:
            raise ValueError("regions must divide representation_dim")
        self.representation_dim = representation_dim
        self.regions = regions
        self.region_width = representation_dim // regions
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(3, 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(1, 3, kernel_size=1),
        )

    def forward(self, branches: torch.Tensor) -> AttentionOutput:
        expected_tail = (3, self.representation_dim)
        if branches.ndim != 3 or tuple(branches.shape[1:]) != expected_tail:
            raise ValueError(
                f"expected [batch, 3, {self.representation_dim}], got {tuple(branches.shape)}"
            )
        regional = branches.reshape(
            branches.shape[0], 3, self.regions, self.region_width
        )
        average_summary = regional.mean(dim=-1)
        maximum_summary = regional.amax(dim=-1)
        regional_weights = torch.sigmoid(
            self.shared_mlp(average_summary) + self.shared_mlp(maximum_summary)
        )
        weights = regional_weights.repeat_interleave(self.region_width, dim=-1)
        weighted_branches = weights * branches
        return AttentionOutput(
            weights=weights,
            regional_weights=regional_weights,
            weighted_branches=weighted_branches,
            fused=weighted_branches.sum(dim=1),
        )
