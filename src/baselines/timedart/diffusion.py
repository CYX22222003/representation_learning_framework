"""Patchwise cosine corruption used only while pretraining TimeDART."""

from __future__ import annotations

import math

import torch
from torch import nn


class PatchDiffusion(nn.Module):
    def __init__(self, steps: int = 1000) -> None:
        super().__init__()
        if steps <= 1:
            raise ValueError("diffusion needs at least two schedule steps")
        self.steps = steps
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
        cumulative = torch.cos(((grid / steps) + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999)
        alpha = 1 - beta
        self.register_buffer("beta", beta, persistent=True)
        self.register_buffer("gamma", alpha.cumprod(dim=0), persistent=True)

    def forward(
        self,
        clean_patches: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if clean_patches.ndim != 3:
            raise ValueError("diffusion input must be [batch*channel, patch, point]")
        if timesteps is None:
            timesteps = torch.randint(
                self.steps, clean_patches.shape[:2], device=clean_patches.device
            )
        if timesteps.shape != clean_patches.shape[:2] or timesteps.dtype != torch.long:
            raise ValueError("diffusion timesteps have the wrong shape or type")
        if timesteps.device != clean_patches.device or (timesteps < 0).any() or (timesteps >= self.steps).any():
            raise ValueError("diffusion timesteps are out of range or on another device")
        if noise is None:
            noise = torch.randn_like(clean_patches)
        if noise.shape != clean_patches.shape or not torch.isfinite(noise).all():
            raise ValueError("diffusion noise must be finite and match patch shape")
        gamma = self.gamma[timesteps].unsqueeze(-1).to(dtype=clean_patches.dtype)
        corrupted = gamma.sqrt() * clean_patches + (1 - gamma).sqrt() * noise
        return corrupted, noise, timesteps
