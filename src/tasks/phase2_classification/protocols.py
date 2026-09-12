from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    sampling: str
    loss: str
    candidate: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PROTOCOLS: dict[str, ProtocolSpec] = {
    "P0": ProtocolSpec("P0", "natural", "hard_ce", False, "untreated reference"),
    "P1U": ProtocolSpec("P1U", "majority_undersampling", "hard_ce", True, "majority-only undersampling"),
    "P1O": ProtocolSpec("P1O", "balanced_oversampling", "hard_ce", True, "fully balanced oversampling"),
    "P2": ProtocolSpec("P2", "natural", "logit_adjusted_ce", True, "train-prior logit adjustment"),
}


def class_counts(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError("labels outside configured class range")
    return np.bincount(y, minlength=n_classes).astype(np.int64)


def class_priors(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    counts = class_counts(labels, n_classes)
    if np.any(counts == 0):
        raise ValueError("logit adjustment requires every class in training data")
    return (counts / counts.sum()).astype(np.float32)


def protocol_sample_indices(labels: np.ndarray, protocol_id: str, seed: int, n_classes: int = 3) -> tuple[np.ndarray, dict[str, object]]:
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol_id}")
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = class_counts(y, n_classes)
    rng = np.random.default_rng(seed)
    spec = PROTOCOLS[protocol_id]
    if spec.sampling == "natural":
        selected = np.arange(len(y), dtype=np.int64)
    elif spec.sampling == "majority_undersampling":
        majority = int(np.argmax(counts))
        if int(np.sum(counts == counts[majority])) != 1:
            raise ValueError("P1U requires a unique majority class")
        minority_counts = np.delete(counts, majority)
        if np.any(minority_counts == 0):
            raise ValueError("P1U requires both minority classes")
        target = int(minority_counts.min())
        parts = []
        for class_id in range(n_classes):
            candidates = np.flatnonzero(y == class_id)
            if class_id == majority:
                candidates = rng.choice(candidates, size=target, replace=False)
            parts.append(np.asarray(candidates, dtype=np.int64))
        selected = np.concatenate(parts)
        selected = selected[rng.permutation(len(selected))]
    elif spec.sampling == "balanced_oversampling":
        target = int(counts.max())
        parts = []
        for class_id in range(n_classes):
            candidates = np.flatnonzero(y == class_id)
            if len(candidates) == 0:
                raise ValueError("P1O requires every class in training data")
            extra = rng.choice(candidates, size=target - len(candidates), replace=True)
            parts.append(np.concatenate([candidates, extra]).astype(np.int64))
        selected = np.concatenate(parts)
        selected = selected[rng.permutation(len(selected))]
    else:  # pragma: no cover
        raise AssertionError(spec.sampling)
    draw_counts = class_counts(y[selected], n_classes)
    audit = {
        **spec.to_dict(), "seed": int(seed), "original_counts": counts.tolist(),
        "draw_counts": draw_counts.tolist(), "draw_count": int(len(selected)),
        "unique_source_count": int(len(np.unique(selected))),
        "duplicate_count": int(len(selected) - len(np.unique(selected))),
    }
    return selected.astype(np.int64), audit


class LogitAdjustedCrossEntropy(nn.Module):
    """Cross-entropy applied to logits + strength * log(training prior)."""

    def __init__(self, priors: np.ndarray, strength: float = 1.0) -> None:
        super().__init__()
        prior = torch.as_tensor(priors, dtype=torch.float32)
        if prior.ndim != 1 or torch.any(prior <= 0) or not torch.isclose(prior.sum(), torch.tensor(1.0)):
            raise ValueError("priors must be a positive 1-D probability vector")
        if strength < 0.0:
            raise ValueError("strength must be non-negative")
        self.register_buffer("log_prior", torch.log(prior))
        self.strength = float(strength)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.strength * self.log_prior, targets)


def make_loss(protocol_id: str, priors: np.ndarray, strength: float = 1.0) -> nn.Module:
    if protocol_id not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol_id}")
    if PROTOCOLS[protocol_id].loss == "hard_ce":
        return nn.CrossEntropyLoss()
    return LogitAdjustedCrossEntropy(priors, strength=strength)
