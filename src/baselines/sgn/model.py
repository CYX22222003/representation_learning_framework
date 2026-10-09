"""Independent SGN-C model matching the approved tensor contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import SGNConfig
from .grouping import LearnableGroupAssignment


class SinusoidalPosition(nn.Module):
    def __init__(self, width: int, length: int) -> None:
        super().__init__()
        position = torch.arange(length, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10_000.0) / width))
        values = torch.zeros(length, width, dtype=torch.float32)
        values[:, 0::2] = torch.sin(position * frequencies)
        values[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("values", values[None], persistent=True)

    def forward(self, length: int) -> torch.Tensor:
        return self.values[:, :length]


class SharedGroupEmbedding(nn.Module):
    def __init__(self, config: SGNConfig) -> None:
        super().__init__()
        self.width = config.embedding_dim
        self.token = nn.Conv1d(1, self.width, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.token.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.position = SinusoidalPosition(self.width, config.seq_len)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, grouped: torch.Tensor) -> torch.Tensor:
        batch, length, groups = grouped.shape
        values = grouped.permute(0, 2, 1).reshape(batch * groups, length, 1)
        embedded = self.token(values.transpose(1, 2)).transpose(1, 2)
        embedded = self.dropout(embedded + self.position(length))
        return embedded.reshape(batch, groups, length, self.width).permute(0, 1, 3, 2).reshape(batch, groups * self.width, length)


class AveragedDepthwiseBank(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, groups: int, kernels: tuple[int, ...]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel, padding=kernel // 2, groups=groups)
            for kernel in kernels
        ])
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([layer(values) for layer in self.layers], dim=-1).mean(dim=-1)


class GroupWindowMixer(nn.Module):
    def __init__(self, config: SGNConfig) -> None:
        super().__init__()
        groups, width, ratio = config.num_groups, config.embedding_dim, config.expansion_ratio
        channels = groups * width
        self.groups, self.width, self.ratio = groups, width, ratio
        self.temporal = nn.Sequential(
            AveragedDepthwiseBank(channels, channels * ratio, channels, config.kernel_sizes),
            nn.GELU(),
            AveragedDepthwiseBank(channels * ratio, channels, channels, config.kernel_sizes),
        )
        self.norm = nn.BatchNorm1d(width)
        self.intra = nn.Sequential(
            nn.Conv1d(channels, channels * ratio, 1, groups=groups), nn.GELU(), nn.Dropout(config.dropout),
            nn.Conv1d(channels * ratio, channels, 1, groups=groups), nn.Dropout(config.dropout),
        )
        self.inter_up = nn.Conv1d(channels, channels * ratio, 1, groups=width)
        self.inter_act = nn.GELU()
        self.inter_drop1 = nn.Dropout(config.dropout)
        self.inter_down = nn.Conv1d(channels * ratio, channels, 1, groups=width)
        self.inter_drop2 = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, windows, period = values.shape
        residual = values
        mixed = values.permute(0, 2, 1, 3).reshape(batch * windows, channels, period)
        mixed = self.temporal(mixed)
        mixed = mixed.reshape(batch * windows, self.groups, self.width, period)
        mixed = self.norm(mixed.reshape(batch * windows * self.groups, self.width, period))
        mixed = mixed.reshape(batch * windows, channels, period)
        mixed = self.intra(mixed).reshape(batch * windows, self.groups, self.width, period)
        mixed = mixed.permute(0, 2, 1, 3).reshape(batch * windows, channels, period)
        mixed = self.inter_drop1(self.inter_up(mixed))
        mixed = self.inter_act(mixed)
        mixed = self.inter_drop2(self.inter_down(mixed))
        mixed = mixed.reshape(batch * windows, self.width, self.groups, period).permute(0, 2, 1, 3)
        mixed = mixed.reshape(batch, windows, channels, period).permute(0, 2, 1, 3)
        return mixed + residual


class ShiftedWindowBlock(nn.Module):
    def __init__(self, mixer: GroupWindowMixer, shifted: bool) -> None:
        super().__init__()
        self.mixer = mixer
        self.shifted = bool(shifted)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.shifted:
            return self.mixer(values)
        batch, channels, windows, period = values.shape
        original = values.reshape(batch, channels, windows * period)
        shifted = torch.roll(original, shifts=-(period // 2), dims=2).reshape_as(values)
        restored = torch.roll(self.mixer(shifted).reshape_as(original), shifts=period // 2, dims=2)
        restored = restored.clone()
        restored[:, :, : period // 2] = original[:, :, : period // 2]
        return restored.reshape_as(values)


class PeriodMerge(nn.Module):
    def __init__(self, config: SGNConfig) -> None:
        super().__init__()
        self.groups, self.width = config.num_groups, config.embedding_dim
        self.reduction = nn.Linear(2 * self.width, self.width)
        self.norm = nn.LayerNorm(self.width)

    def _reduce(self, paired: torch.Tensor) -> torch.Tensor:
        # [B,G,2D,N,P] -> [B,G,D,N,P]
        reduced = self.reduction(paired.permute(0, 1, 3, 4, 2))
        reduced = self.norm(reduced)
        return reduced.permute(0, 1, 4, 2, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, windows, period = values.shape
        if windows <= 1:
            raise ValueError("cannot merge a single period window")
        grouped = values.reshape(batch, self.groups, self.width, windows, period)
        if windows <= 4:
            pairs = [torch.cat((grouped[:, :, :, index], grouped[:, :, :, index + 1]), dim=2) for index in range(windows - 1)]
            paired = torch.stack(pairs, dim=3)
            output = self._reduce(paired)
        else:
            pair_count = windows // 2
            even, odd = grouped[:, :, :, 0 : 2 * pair_count : 2], grouped[:, :, :, 1 : 2 * pair_count : 2]
            output = self._reduce(torch.cat((even, odd), dim=2))
            if windows % 2:
                output = torch.cat((output, grouped[:, :, :, -1:]), dim=3)
        return output.reshape(batch, self.groups * self.width, output.shape[3], period)


class SGNStage(nn.Module):
    def __init__(self, config: SGNConfig, depth: int, merge: bool) -> None:
        super().__init__()
        mixer = GroupWindowMixer(config)
        self.blocks = nn.ModuleList([ShiftedWindowBlock(mixer, shifted=index % 2 == 1) for index in range(depth)])
        self.merge = PeriodMerge(config) if merge else None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            values = block(values)
        return self.merge(values) if self.merge is not None else values


class SGNClassifier(nn.Module):
    def __init__(self, initial_logits: np.ndarray, config: SGNConfig | None = None) -> None:
        super().__init__()
        self.config = config or SGNConfig()
        self.assignment = LearnableGroupAssignment(
            initial_logits,
            temperature_initial=self.config.temperature_initial,
            temperature_minimum=self.config.temperature_minimum,
            temperature_decay=self.config.temperature_decay,
        )
        self.embedding = SharedGroupEmbedding(self.config)
        self.stages = nn.ModuleList([
            SGNStage(self.config, depth, merge=index < len(self.config.stage_depths) - 1)
            for index, depth in enumerate(self.config.stage_depths)
        ])
        self.norm = nn.LayerNorm(self.config.pooled_dim)
        self.head = nn.Linear(self.config.pooled_dim, self.config.num_classes)

    def forward(self, values: torch.Tensor, *, return_diagnostics: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if values.ndim != 3 or tuple(values.shape[1:]) != (64, 5):
            raise ValueError("SGN-C input must be [B,64,5]")
        assignment = self.assignment.assignment()
        grouped = torch.einsum("btc,cg->btg", values, assignment)
        output = self.embedding(grouped)
        output = output.reshape(values.shape[0], self.config.pooled_dim, self.config.seq_len // self.config.period, self.config.period)
        for stage in self.stages:
            output = stage(output)
        positions = output.reshape(values.shape[0], self.config.pooled_dim, -1).transpose(1, 2)
        pooled = self.norm(positions).mean(dim=1)
        logits = self.head(pooled)
        if not return_diagnostics:
            return logits
        regularizer = self.assignment.similarity_regularizer(values, assignment)
        probabilities = torch.softmax(self.assignment.assignment_logits / self.assignment.temperature, dim=-1)
        diagnostics = {
            "similarity_regularizer": regularizer,
            "assignment": assignment,
            "assignment_probabilities": probabilities,
            "assignment_entropy": -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1).mean(),
            "group_mass": assignment.sum(dim=0),
            "temperature": self.assignment.temperature.clone(),
            "training_step": self.assignment.training_step.clone(),
            "pooled": pooled,
        }
        return logits, diagnostics

    @torch.no_grad()
    def advance_temperature(self) -> None:
        self.assignment.advance_temperature()
