"""Training-only BDC/K-means initialization and learnable assignments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_NAMES = ("open", "high", "low", "close", "volume")


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def contract_stratified_indices(condition_ids: np.ndarray, decision_dates: np.ndarray, sample_size: int = 2000) -> np.ndarray:
    """Select deterministic chronological quantiles with proportional contract allocation."""
    conditions = np.asarray(condition_ids)
    dates = np.asarray(decision_dates, dtype=np.int64)
    if conditions.ndim != 1 or dates.shape != conditions.shape:
        raise ValueError("condition/date arrays must be aligned 1-D vectors")
    if sample_size <= 0 or sample_size > len(conditions):
        raise ValueError("invalid BDC sample size")
    unique, inverse, counts = np.unique(conditions, return_inverse=True, return_counts=True)
    if sample_size < len(unique):
        raise ValueError("sample must contain at least one row per contract")
    raw = counts.astype(np.float64) * sample_size / len(conditions)
    allocation = np.maximum(1, np.floor(raw).astype(np.int64))
    allocation = np.minimum(allocation, counts)
    while int(allocation.sum()) < sample_size:
        candidates = np.flatnonzero(allocation < counts)
        score = raw[candidates] - allocation[candidates]
        chosen = candidates[np.lexsort((candidates, -score))[0]]
        allocation[chosen] += 1
    while int(allocation.sum()) > sample_size:
        candidates = np.flatnonzero(allocation > 1)
        score = allocation[candidates] - raw[candidates]
        chosen = candidates[np.lexsort((candidates, -score))[0]]
        allocation[chosen] -= 1
    selected: list[np.ndarray] = []
    for contract_index in range(len(unique)):
        rows = np.flatnonzero(inverse == contract_index)
        rows = rows[np.lexsort((rows, dates[rows]))]
        count = int(allocation[contract_index])
        positions = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
        selected.append(rows[positions])
    result = np.sort(np.concatenate(selected).astype(np.int64))
    if len(result) != sample_size or len(np.unique(result)) != sample_size:
        raise RuntimeError("BDC sampler did not return unique requested rows")
    return result


def _double_centered_distance(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float64)
    squared = np.sum(values * values, axis=1, keepdims=True)
    distances = np.sqrt(np.maximum(squared + squared.T - 2.0 * values @ values.T, 0.0))
    return distances - distances.mean(axis=0, keepdims=True) - distances.mean(axis=1, keepdims=True) + distances.mean()


def brownian_distance_correlation(sequences: np.ndarray) -> np.ndarray:
    """Return source-layout distance correlation for [N,L,C] observations."""
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 5 or not np.isfinite(values).all():
        raise ValueError("BDC expects finite [N,L,5] sequences")
    centered = [_double_centered_distance(values[:, :, channel]) for channel in range(values.shape[2])]
    self_terms = np.asarray([np.mean(matrix * matrix) for matrix in centered])
    if np.any(self_terms <= 0):
        raise ValueError("BDC encountered a constant variable")
    result = np.empty((values.shape[2], values.shape[2]), dtype=np.float64)
    for left in range(values.shape[2]):
        for right in range(left, values.shape[2]):
            value = np.mean(centered[left] * centered[right]) / math.sqrt(self_terms[left] * self_terms[right])
            result[left, right] = result[right, left] = value
    np.fill_diagonal(result, 1.0)
    if not np.isfinite(result).all() or not np.allclose(result, result.T, atol=1e-12):
        raise RuntimeError("invalid BDC matrix")
    return result


def deterministic_kmeans(points: np.ndarray, num_groups: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic Lloyd fit with farthest-point starts and canonical labels."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or num_groups != 2 or len(values) < num_groups:
        raise ValueError("approved initializer requires two groups")
    pairwise = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=-1)
    first, second = np.unravel_index(np.argmax(pairwise), pairwise.shape)
    centroids = values[[first, second]].copy()
    labels = np.full(len(values), -1, dtype=np.int64)
    for _ in range(100):
        distances = np.sum((values[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        updated = np.argmin(distances, axis=1).astype(np.int64)
        if np.array_equal(updated, labels):
            break
        labels = updated
        if len(np.unique(labels)) != num_groups:
            raise RuntimeError("K-means produced an empty group")
        centroids = np.stack([values[labels == group].mean(axis=0) for group in range(num_groups)])
    order = sorted(range(num_groups), key=lambda group: (int(np.sum(labels == group)), int(np.min(np.flatnonzero(labels == group)))))
    remap = np.empty(num_groups, dtype=np.int64)
    for new, old in enumerate(order):
        remap[old] = new
    labels = remap[labels]
    centroids = np.stack([values[labels == group].mean(axis=0) for group in range(num_groups)])
    return labels, centroids


def period_candidates(sequences: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (64, 5):
        raise ValueError("period diagnostic expects [N,64,5]")
    amplitude = np.abs(np.fft.rfft(values, axis=1)).mean(axis=(0, 2))
    amplitude[:2] = -np.inf  # DC and the full-sequence period are excluded.
    bins = np.argsort(amplitude)[-top_k:][::-1].astype(np.int64)
    periods = np.asarray([int(round(values.shape[1] / int(index))) for index in bins], dtype=np.int64)
    return bins, periods


@dataclass(frozen=True)
class GroupInitialization:
    sample_indices: np.ndarray
    bdc_matrix: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    initial_logits: np.ndarray
    fft_bins: np.ndarray
    fft_periods: np.ndarray

    def to_manifest(self) -> dict[str, object]:
        return {
            "sample_size": int(len(self.sample_indices)),
            "sample_indices_sha256": _hash_array(self.sample_indices),
            "bdc_sha256": _hash_array(self.bdc_matrix),
            "labels": self.labels.tolist(),
            "centroids_sha256": _hash_array(self.centroids),
            "initial_logits_sha256": _hash_array(self.initial_logits),
            "fft_bins": self.fft_bins.tolist(),
            "fft_periods": self.fft_periods.tolist(),
            "approved_period_present": bool(16 in self.fft_periods),
        }


def build_group_initialization(
    sequences: np.ndarray,
    condition_ids: np.ndarray,
    decision_dates: np.ndarray,
    *,
    sample_size: int = 2000,
    seed: int = 0,
) -> GroupInitialization:
    indices = contract_stratified_indices(condition_ids, decision_dates, sample_size)
    bdc = brownian_distance_correlation(np.asarray(sequences)[indices])
    labels, centroids = deterministic_kmeans(bdc, 2)
    one_hot = np.eye(2, dtype=np.float32)[labels]
    logits = np.clip(one_hot + np.random.default_rng(seed).normal(0.0, 0.1, one_hot.shape), 0.0, 1.0).astype(np.float32)
    bins, periods = period_candidates(np.asarray(sequences))
    if 16 not in periods:
        raise ValueError("approved period 16 is absent from the training-only candidates")
    return GroupInitialization(indices, bdc, labels, centroids, logits, bins, periods)


class LearnableGroupAssignment(nn.Module):
    def __init__(self, initial_logits: np.ndarray, *, temperature_initial: float = 1.0,
                 temperature_minimum: float = 0.1, temperature_decay: float = 3e-4) -> None:
        super().__init__()
        logits = torch.as_tensor(initial_logits, dtype=torch.float32)
        if tuple(logits.shape) != (5, 2) or not torch.isfinite(logits).all():
            raise ValueError("initial assignment logits must be finite [5,2]")
        self.assignment_logits = nn.Parameter(logits.clone())
        self.temperature_initial = float(temperature_initial)
        self.temperature_minimum = float(temperature_minimum)
        self.temperature_decay = float(temperature_decay)
        self.register_buffer("temperature", torch.tensor(self.temperature_initial, dtype=torch.float32))
        self.register_buffer("training_step", torch.tensor(0, dtype=torch.long))

    def assignment(self) -> torch.Tensor:
        if self.training:
            uniform = torch.rand_like(self.assignment_logits).clamp_(1e-9, 1.0 - 1e-9)
            noise = -torch.log(-torch.log(uniform))
            return F.softmax((self.assignment_logits + noise) / self.temperature, dim=-1)
        probabilities = F.softmax(self.assignment_logits / self.temperature_minimum, dim=-1)
        return F.one_hot(probabilities.argmax(dim=-1), num_classes=2).to(probabilities.dtype)

    @torch.no_grad()
    def advance_temperature(self) -> None:
        self.training_step.add_(1)
        value = max(self.temperature_minimum, self.temperature_initial * math.exp(-self.temperature_decay * int(self.training_step)))
        self.temperature.fill_(value)

    @staticmethod
    def similarity_matrix(values: torch.Tensor) -> torch.Tensor:
        channels = F.normalize(values.permute(2, 0, 1).reshape(values.shape[2], -1), dim=1)
        return channels @ channels.T

    def similarity_regularizer(self, values: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        similarity = (self.similarity_matrix(values.detach()) + 1.0) / 2.0
        difference = assignment[:, None, :] - assignment[None, :, :]
        return (similarity[..., None] * difference.square()).sum() / (values.shape[2] ** 2)
