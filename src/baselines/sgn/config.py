"""Frozen SGN-C numerical contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SGNConfig:
    seq_len: int = 64
    input_dim: int = 5
    num_classes: int = 3
    num_groups: int = 2
    embedding_dim: int = 64
    period: int = 16
    stage_depths: tuple[int, ...] = (2, 2, 2, 1)
    num_kernels: int = 7
    expansion_ratio: int = 2
    dropout: float = 0.1
    temperature_initial: float = 1.0
    temperature_minimum: float = 0.1
    temperature_decay: float = 3e-4
    grouping_regularizer_beta: float = 0.1

    def __post_init__(self) -> None:
        if self.seq_len != 64 or self.input_dim != 5 or self.num_classes != 3:
            raise ValueError("SGN-C requires the frozen [64,5] -> 3 contract")
        if (self.num_groups, self.embedding_dim, self.period) != (2, 64, 16):
            raise ValueError("SGN-C requires G=2, D=64, P=16")
        if self.stage_depths != (2, 2, 2, 1):
            raise ValueError("SGN-C requires stage depths [2,2,2,1]")
        if (self.num_kernels, self.expansion_ratio, self.dropout) != (7, 2, 0.1):
            raise ValueError("SGN-C kernel/ratio/dropout contract differs")
        if (self.temperature_initial, self.temperature_minimum, self.temperature_decay) != (1.0, 0.1, 3e-4):
            raise ValueError("SGN-C temperature contract differs")
        if self.grouping_regularizer_beta != 0.1:
            raise ValueError("SGN-C beta must equal 0.1")
        if self.seq_len % self.period:
            raise ValueError("approved SGN-C period must divide the sequence exactly")
        count = self.seq_len // self.period
        for index in range(len(self.stage_depths) - 1):
            if count <= 1:
                raise ValueError(f"stage {index} would merge a single window")
            count = count - 1 if count <= 4 else (count + 1) // 2
        if count != 1:
            raise ValueError("approved hierarchy must terminate at one period window")

    @property
    def pooled_dim(self) -> int:
        return self.num_groups * self.embedding_dim

    @property
    def kernel_sizes(self) -> tuple[int, ...]:
        return tuple(2 * index + 1 for index in range(self.num_kernels))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage_depths"] = list(self.stage_depths)
        payload["kernel_sizes"] = list(self.kernel_sizes)
        payload["pooled_dim"] = self.pooled_dim
        return payload
