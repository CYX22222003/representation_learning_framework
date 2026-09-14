from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from aggregation.aggregator import RepresentationAggregator
from tasks.price_prediction import PriceRegressor
from tasks.trend_classification import TrendClassifier
from tasks.volatility_prediction import VolatilityRegressor


DECODER_IDS = ("D0", "D1", "D2", "D3", "D4")


class ResidualBlock(nn.Module):
    def __init__(self, width: int = 128, expansion: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*expansion), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(width*expansion, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TaskReadout(nn.Module):
    def __init__(self, width: int, task: str) -> None:
        super().__init__()
        outputs = 3 if task == "trend_classification" else 1
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width//2), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(width//2, outputs))
        self.nonnegative = task == "volatility_prediction"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.net(x)
        return torch.nn.functional.softplus(value) if self.nonnegative else value


def _phase1_head(width: int, task: str) -> nn.Module:
    if task == "price_prediction":
        return PriceRegressor(width, 128)
    if task == "volatility_prediction":
        return nn.Sequential(VolatilityRegressor(width, 128), nn.Softplus())
    if task == "trend_classification":
        return TrendClassifier(width, 128, 3)
    raise ValueError(f"unsupported task: {task}")


class SinusoidalPosition(nn.Module):
    def __init__(self, width: int, max_length: int) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0))/width))
        values = torch.zeros(max_length, width)
        values[:, 0::2] = torch.sin(position * div)
        values[:, 1::2] = torch.cos(position * div[:values[:, 1::2].shape[1]])
        self.register_buffer("values", values, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.values[:x.shape[1]]


class DecoderModel(nn.Module):
    def __init__(self, decoder_id: str, branch_dims: Mapping[str, int], task: str, context_length: int = 8) -> None:
        super().__init__()
        if decoder_id not in DECODER_IDS:
            raise ValueError(f"decoder_id must be one of {DECODER_IDS}")
        self.decoder_id, self.branch_dims, self.task = decoder_id, dict(branch_dims), task
        total = sum(self.branch_dims.values())
        if decoder_id == "D0":
            self.body, self.head = nn.Identity(), _phase1_head(total, task)
        elif decoder_id == "D1":
            self.projectors = nn.ModuleDict({n: nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.LayerNorm(64)) for n,d in self.branch_dims.items()})
            self.body = nn.Sequential(nn.Linear(64*len(self.branch_dims), 128), nn.LayerNorm(128),
                                      ResidualBlock(), ResidualBlock())
            self.head = TaskReadout(128, task)
        elif decoder_id == "D2":
            self.aggregator = RepresentationAggregator(self.branch_dims, out_dim=128, mode="gated")
            self.head = _phase1_head(128, task)
        elif decoder_id == "D3":
            self.body = nn.Sequential(nn.Linear(total, 128), nn.LayerNorm(128), nn.Dropout(0.1))
            self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True)
            self.head = TaskReadout(128, task)
        else:
            self.body = nn.Sequential(nn.Linear(total, 96), nn.LayerNorm(96))
            self.position = SinusoidalPosition(96, context_length)
            layer = nn.TransformerEncoderLayer(96, 4, 192, 0.1, activation="gelu", batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = TaskReadout(96, task)

    def _split(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        parts, offset = {}, 0
        for name, width in self.branch_dims.items():
            parts[name] = x[..., offset:offset+width]
            offset += width
        return parts

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.decoder_id == "D0":
            return self.head(x), None
        if self.decoder_id == "D1":
            projected = [self.projectors[n](v) for n,v in self._split(x).items()]
            return self.head(self.body(torch.cat(projected, dim=-1))), None
        if self.decoder_id == "D2":
            embedding, weights = self.aggregator(self._split(x))
            return self.head(embedding), weights
        if self.decoder_id == "D3":
            _, (hidden, _) = self.lstm(self.body(x))
            return self.head(hidden[-1]), None
        encoded = self.position(self.body(x))
        mask = nn.Transformer.generate_square_subsequent_mask(encoded.shape[1], device=encoded.device)
        return self.head(self.transformer(encoded, mask=mask)[:, -1]), None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)[0]


def make_decoder(decoder_id: str, branch_dims: Mapping[str, int], task: str, context_length: int = 8) -> DecoderModel:
    return DecoderModel(decoder_id, branch_dims, task, context_length)
