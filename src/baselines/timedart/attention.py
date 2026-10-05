"""Causal encoder and self-only denoising blocks for original TimeDART."""

from __future__ import annotations

import torch
from torch import nn

from .config import TimeDARTConfig


def causal_mask(length: int, *, device: torch.device | None = None) -> torch.Tensor:
    if length <= 0:
        raise ValueError("mask length must be positive")
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


def self_only_mask(length: int, *, device: torch.device | None = None) -> torch.Tensor:
    if length <= 0:
        raise ValueError("mask length must be positive")
    return ~torch.eye(length, dtype=torch.bool, device=device)


class EncoderBlock(nn.Module):
    def __init__(self, config: TimeDARTConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.norm_feed_forward = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        attended, _ = self.attention(
            states, states, states, attn_mask=mask, need_weights=False
        )
        states = self.norm_attention(states + self.dropout(attended))
        return self.norm_feed_forward(
            states + self.dropout(self.feed_forward(states))
        )


class PatchTransformer(nn.Module):
    def __init__(self, config: TimeDARTConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.encoder_layers))
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, states: torch.Tensor, *, causal: bool) -> torch.Tensor:
        mask = causal_mask(states.shape[1], device=states.device) if causal else None
        for block in self.blocks:
            states = block(states, mask)
        return self.final_norm(states)


class DecoderBlock(nn.Module):
    def __init__(self, config: TimeDARTConfig) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(config.d_model)
        self.norm_cross = nn.LayerNorm(config.d_model)
        self.norm_feed_forward = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, noisy_states: torch.Tensor, clean_history: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.self_attention(
            noisy_states, noisy_states, noisy_states,
            attn_mask=mask, need_weights=False,
        )
        noisy_states = self.norm_self(noisy_states + self.dropout(attended))
        attended, _ = self.cross_attention(
            noisy_states, clean_history, clean_history,
            attn_mask=mask, need_weights=False,
        )
        noisy_states = self.norm_cross(noisy_states + self.dropout(attended))
        return self.norm_feed_forward(
            noisy_states + self.dropout(self.feed_forward(noisy_states))
        )


class PatchDenoiser(nn.Module):
    def __init__(self, config: TimeDARTConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.decoder_layers))
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, noisy_states: torch.Tensor, clean_history: torch.Tensor) -> torch.Tensor:
        if noisy_states.shape != clean_history.shape:
            raise ValueError("denoising query and history states must align")
        mask = self_only_mask(noisy_states.shape[1], device=noisy_states.device)
        for block in self.blocks:
            noisy_states = block(noisy_states, clean_history, mask)
        return self.final_norm(noisy_states)
