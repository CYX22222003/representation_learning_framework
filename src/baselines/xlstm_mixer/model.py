"""Minimal Phase 6.9 xLSTM-Mixer adapter.

This implementation is independently authored from the paper and the pinned
official source audit. It intentionally excludes upstream data loaders,
Lightning orchestration, hyperparameter search, and checkpoint selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .backend import XLSTMBackend, build_xlstm_stack
from .config import (
    CHECKPOINT_SCHEMA,
    METHOD_ID,
    SOURCE_CONTRACT,
    XLSTMMixerConfig,
)
from .normalization import NonAffineRevIN


@dataclass(frozen=True)
class XLSTMMixerTrace:
    """Intermediate tensors exposed only for architecture/replay validation."""

    normalized_input: torch.Tensor
    preliminary_forecast: torch.Tensor
    embedded_variates: torch.Tensor
    tokens_with_initial: torch.Tensor
    reversed_tokens: torch.Tensor
    forward_view: torch.Tensor
    reverse_view: torch.Tensor
    mixed_variates: torch.Tensor
    normalized_forecast: torch.Tensor
    forecast: torch.Tensor


class XLSTMMixer(nn.Module):
    """Approved ``[B,64,5] -> [B,8,5]`` xLSTM-Mixer complete system.

    ``backend`` is an execution setting rather than an architecture choice.
    Use ``vanilla`` for bounded CPU admission and ``cuda`` for the admitted
    experiment runtime. Checkpoint metadata is backend-independent.
    """

    def __init__(
        self,
        config: XLSTMMixerConfig | None = None,
        *,
        backend: XLSTMBackend = "vanilla",
    ) -> None:
        super().__init__()
        self.config = config or XLSTMMixerConfig()
        if backend not in ("vanilla", "cuda"):
            raise ValueError("backend must be 'vanilla' or 'cuda'")
        self.backend: XLSTMBackend = backend

        self.revin = NonAffineRevIN(
            self.config.num_variates, epsilon=self.config.revin_epsilon
        )
        self.time_projection = nn.Linear(self.config.context_length, self.config.horizon)
        self.up_projection = nn.Linear(self.config.horizon, self.config.embedding_dim)
        self.initial_token = nn.Parameter(
            torch.empty(self.config.num_initial_tokens, self.config.embedding_dim)
        )
        nn.init.normal_(self.initial_token, mean=0.0, std=0.01)
        self.slstm_stack = build_xlstm_stack(self.config, backend)
        self.output_projection = nn.Linear(
            2 * self.config.embedding_dim, self.config.horizon
        )

    @staticmethod
    def reverse_latent_features(tokens: torch.Tensor) -> torch.Tensor:
        """Released ``FULL`` behavior: reverse width, not token order."""

        if tokens.ndim != 3:
            raise ValueError("view reversal expects [batch, tokens, width]")
        return torch.flip(tokens, dims=(-1,))

    def forecast_with_trace(self, x: torch.Tensor) -> XLSTMMixerTrace:
        self._validate_input(x)
        normalized, statistics = self.revin.normalize(x)

        last = normalized[:, -1:, :].detach()
        centered = normalized - last
        preliminary = self.time_projection(centered.transpose(1, 2)).transpose(1, 2)
        preliminary = preliminary + last

        embedded = self.up_projection(preliminary.transpose(1, 2))
        initial = self.initial_token.unsqueeze(0).expand(embedded.shape[0], -1, -1)
        tokens = torch.cat((initial, embedded), dim=1)
        reversed_tokens = self.reverse_latent_features(tokens)

        # Preserve the pinned wrapper's call order because dropout consumes RNG:
        # the reverse view is evaluated before the ordinary view.
        reverse_view = self.slstm_stack(reversed_tokens)
        forward_view = self.slstm_stack(tokens)
        if reverse_view.shape != tokens.shape or forward_view.shape != tokens.shape:
            raise RuntimeError("sLSTM stack changed the token tensor shape")

        mixed_with_initial = torch.cat((forward_view, reverse_view), dim=-1)
        mixed_variates = mixed_with_initial[:, self.config.num_initial_tokens :, :]
        normalized_forecast = self.output_projection(mixed_variates).transpose(1, 2)
        forecast = self.revin.denormalize(normalized_forecast, statistics)
        if forecast.shape != (
            x.shape[0],
            self.config.horizon,
            self.config.num_variates,
        ):
            raise RuntimeError("xLSTM-Mixer produced the wrong output shape")
        if not torch.isfinite(forecast).all():
            raise FloatingPointError("xLSTM-Mixer produced non-finite forecasts")

        return XLSTMMixerTrace(
            normalized_input=normalized,
            preliminary_forecast=preliminary,
            embedded_variates=embedded,
            tokens_with_initial=tokens,
            reversed_tokens=reversed_tokens,
            forward_view=forward_view,
            reverse_view=reverse_view,
            mixed_variates=mixed_variates,
            normalized_forecast=normalized_forecast,
            forecast=forecast,
        )

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecast_with_trace(x).forecast

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forecast(x)

    def headline_close(self, full_path: torch.Tensor) -> torch.Tensor:
        """Extract the frozen Phase 6.9 endpoint ``close[t+8]``."""

        expected = (self.config.horizon, self.config.num_variates)
        if full_path.ndim != 3 or tuple(full_path.shape[1:]) != expected:
            raise ValueError(f"expected full paths shaped [batch,{expected[0]},{expected[1]}]")
        close_index = self.config.channel_order.index("close")
        return full_path[:, self.config.horizon - 1, close_index]

    def full_path_l1_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Unweighted mean L1 over all future hours and accepted-unit channels."""

        expected = (self.config.horizon, self.config.num_variates)
        if prediction.ndim != 3 or tuple(prediction.shape[1:]) != expected:
            raise ValueError("prediction has the wrong full-path shape")
        if target.shape != prediction.shape:
            raise ValueError("target must have the same full-path shape as prediction")
        if not torch.isfinite(target).all():
            raise ValueError("full-path targets must be finite")
        loss = F.l1_loss(prediction, target, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("xLSTM-Mixer full-path loss is non-finite")
        return loss

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "method_id": METHOD_ID,
            "architecture": self.config.to_dict(),
            "architecture_sha256": self.config.sha256,
            "source_contract": SOURCE_CONTRACT.to_dict(),
            "source_contract_sha256": SOURCE_CONTRACT.sha256,
        }

    def get_extra_state(self) -> dict[str, Any]:
        return self.checkpoint_metadata()

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.validate_checkpoint_metadata(state)

    def validate_checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            raise RuntimeError("xLSTM-Mixer checkpoint metadata is missing")
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
            raise RuntimeError("xLSTM-Mixer checkpoint schema mismatch")
        if metadata.get("method_id") != METHOD_ID:
            raise RuntimeError("xLSTM-Mixer method identity mismatch")
        if metadata.get("architecture") != self.config.to_dict():
            raise RuntimeError("xLSTM-Mixer architecture configuration mismatch")
        if metadata.get("architecture_sha256") != self.config.sha256:
            raise RuntimeError("xLSTM-Mixer architecture hash mismatch")
        if metadata.get("source_contract") != SOURCE_CONTRACT.to_dict():
            raise RuntimeError("xLSTM-Mixer source contract mismatch")
        if metadata.get("source_contract_sha256") != SOURCE_CONTRACT.sha256:
            raise RuntimeError("xLSTM-Mixer source contract hash mismatch")

    def _validate_input(self, x: torch.Tensor) -> None:
        expected = (self.config.context_length, self.config.num_variates)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"expected contexts shaped [batch,{expected[0]},{expected[1]}]")
        if not x.is_floating_point():
            raise TypeError("xLSTM-Mixer contexts must use a floating dtype")
        if not torch.isfinite(x).all():
            raise ValueError("xLSTM-Mixer contexts must be finite")
