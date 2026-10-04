from __future__ import annotations

import torch
import torch.nn.functional as F


def positive_pair_indices(batch_size: int, device: torch.device | str) -> torch.Tensor:
    if batch_size < 2:
        raise ValueError("NT-Xent requires at least two samples")
    rows = torch.arange(2 * batch_size, device=device)
    return (rows + batch_size) % (2 * batch_size)


def negative_pair_mask(batch_size: int, device: torch.device | str) -> torch.Tensor:
    """Return a mask that excludes each anchor and its cross-view positive."""

    positives = positive_pair_indices(batch_size, device)
    rows = torch.arange(2 * batch_size, device=device)
    mask = torch.ones(
        (2 * batch_size, 2 * batch_size), dtype=torch.bool, device=device
    )
    mask[rows, rows] = False
    mask[rows, positives] = False
    return mask


def symmetric_nt_xent(
    left: torch.Tensor, right: torch.Tensor, *, temperature: float = 0.15
) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("NT-Xent inputs must have the same [batch, features] shape")
    if not left.is_floating_point() or not right.is_floating_point():
        raise TypeError("NT-Xent inputs must be floating-point tensors")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    batch_size = left.shape[0]
    positives = positive_pair_indices(batch_size, left.device)
    mask = negative_pair_mask(batch_size, left.device)
    representations = torch.cat(
        (F.normalize(left, dim=1), F.normalize(right, dim=1)), dim=0
    )
    similarities = representations @ representations.transpose(0, 1)
    rows = torch.arange(2 * batch_size, device=left.device)
    positive_logits = similarities[rows, positives].unsqueeze(1)
    negative_logits = similarities[mask].reshape(2 * batch_size, -1)
    logits = torch.cat((positive_logits, negative_logits), dim=1) / temperature
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=left.device)
    return F.cross_entropy(logits, labels)


def mean_sample_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over samples of the coordinatewise L1 sum from paper Equation 6."""

    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("mapping tensors must have the same [batch, features] shape")
    return (prediction - target).abs().sum(dim=1).mean()


def two_way_mapping_loss(
    mapped_fourier: torch.Tensor,
    fourier_target: torch.Tensor,
    mapped_wavelet: torch.Tensor,
    wavelet_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fourier = mean_sample_l1(mapped_fourier, fourier_target)
    wavelet = mean_sample_l1(mapped_wavelet, wavelet_target)
    return fourier, wavelet, fourier + wavelet
