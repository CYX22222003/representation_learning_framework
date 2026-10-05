"""Original TimeDART pretrainer and approved decoder-free frozen extractor.

This project implementation is independently authored from the ICML paper
and attributed behavioral audit. No official source code is vendored here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention import PatchDenoiser, PatchTransformer
from .config import TimeDARTConfig
from .diffusion import PatchDiffusion
from .patching import SinusoidalPositions, channelwise_patches, normalize_instance, unpatch_channels


@dataclass(frozen=True)
class TimeDARTPretrainOutput:
    reconstruction: torch.Tensor
    loss: torch.Tensor
    timesteps: torch.Tensor


class TimeDARTEncoder(nn.Module):
    """Shared channelwise patch encoder; forward emits the frozen 170-vector."""

    def __init__(self, config: TimeDARTConfig | None = None) -> None:
        super().__init__()
        self.config = config or TimeDARTConfig()
        self.patch_embedding = nn.Linear(self.config.patch_len, self.config.d_model)
        self.positions = SinusoidalPositions(
            length=self.config.patch_count,
            width=self.config.d_model,
            dropout=self.config.dropout,
        )
        self.transformer = PatchTransformer(self.config)

    def embed_patches(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (
            self.config.patch_count, self.config.patch_len
        ):
            raise ValueError("patch embedding received the wrong shape")
        return self.patch_embedding(patches)

    def contextualize(self, embeddings: torch.Tensor, *, causal: bool) -> torch.Tensor:
        return self.transformer(self.positions(embeddings), causal=causal)

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        normalized, means, stds = normalize_instance(
            x, epsilon=self.config.instance_norm_epsilon
        )
        patches = channelwise_patches(normalized, self.config)
        # The source downstream forecasting path uses the clean unshifted
        # sequence and removes the causal attention mask.
        states = self.contextualize(self.embed_patches(patches), causal=False)
        batch = x.shape[0]
        pooled = states.amax(dim=1).reshape(batch, self.config.channels * self.config.d_model)
        features = torch.cat((pooled, means[:, 0, :], stds[:, 0, :]), dim=-1)
        if features.shape != (batch, self.config.representation_width):
            raise RuntimeError("TimeDART extractor returned the wrong width")
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract(x)


class TimeDARTPretrainer(nn.Module):
    """Target-free patch reconstruction with causal clean history."""

    def __init__(self, config: TimeDARTConfig | None = None) -> None:
        super().__init__()
        self.config = config or TimeDARTConfig()
        self.encoder = TimeDARTEncoder(self.config)
        self.start_token = nn.Parameter(torch.randn(1, 1, self.config.d_model))
        self.diffusion = PatchDiffusion(self.config.diffusion_steps)
        self.denoiser = PatchDenoiser(self.config)
        self.patch_projector = nn.Linear(self.config.d_model, self.config.patch_len)
        self.projector_dropout = nn.Dropout(self.config.projector_dropout)

    def _clean_history(
        self, clean_patches: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.encoder.embed_patches(clean_patches)
        start = self.start_token.expand(embedded.shape[0], -1, -1)
        shifted = torch.cat((start, embedded[:, :-1]), dim=1)
        return self.encoder.contextualize(shifted, causal=True)

    def clean_history_states(self, x: torch.Tensor) -> torch.Tensor:
        normalized, _, _ = normalize_instance(
            x, epsilon=self.config.instance_norm_epsilon
        )
        patches = channelwise_patches(normalized, self.config)
        return self._clean_history(patches)

    def forward(
        self,
        x: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> TimeDARTPretrainOutput:
        normalized, means, stds = normalize_instance(
            x, epsilon=self.config.instance_norm_epsilon
        )
        patches = channelwise_patches(normalized, self.config)
        history = self._clean_history(patches)
        noisy_patches, _, sampled_steps = self.diffusion(
            patches, timesteps=timesteps, noise=noise
        )
        noisy_embeddings = self.encoder.positions(self.encoder.embed_patches(noisy_patches))
        decoded = self.denoiser(noisy_embeddings, history)
        reconstructed_patches = self.projector_dropout(self.patch_projector(decoded))
        reconstructed_normalized = unpatch_channels(reconstructed_patches, self.config)
        reconstruction = reconstructed_normalized * stds + means
        loss = F.mse_loss(reconstruction, x)
        if not torch.isfinite(loss):
            raise FloatingPointError("TimeDART reconstruction loss is non-finite")
        return TimeDARTPretrainOutput(reconstruction, loss, sampled_steps)

    def frozen_encoder(self) -> TimeDARTEncoder:
        """Export only the decoder-free representation path at the selected epoch."""
        encoder = deepcopy(self.encoder).eval()
        encoder.requires_grad_(False)
        return encoder
