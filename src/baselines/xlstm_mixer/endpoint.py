"""Owner-approved direct close[t+8h] adaptation, distinct from XM-MV8."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .backend import XLSTMBackend, build_xlstm_stack
from .config import SOURCE_CONTRACT, XLSTMMixerConfig
from .normalization import NonAffineRevIN

METHOD_ID = "XM-C8"
METHOD = "xm_c8"
MODEL_SCHEMA = "phase6-9-xlstm-mixer-endpoint-model-v1"


def architecture_manifest() -> dict[str, Any]:
    """Forecast count is one; the physical lead time is eight hours."""
    core = XLSTMMixerConfig().to_dict()
    core.pop("horizon")
    core.pop("full_path_loss")
    return {
        **core,
        "method_id": METHOD_ID,
        "forecast_steps": 1,
        "forecast_horizon_hours": 8,
        "target_channel": "close",
        "output_shape": "[B,1]",
        "loss": "mse_mean_raw_probability",
        "output_transform": "inverse_non_affine_revin_close_unclipped",
        "auxiliary_supervision": False,
        "time_projection": [64, 1],
        "up_projection": [1, 128],
        "output_projection": [256, 1],
    }


class XLSTMMixerEndpoint(nn.Module):
    """Five historical variates mixed into a single direct h8 close forecast.

    The shared NLinear/up projections produce one preliminary endpoint per
    historical variate. Only the mixed close token is decoded and inverse-
    normalized. There are no future OHLCV path outputs or auxiliary losses.
    RevIN's inverse stays unclipped, as in the audited forecasting core;
    out-of-range probabilities are diagnostics, not repaired predictions.
    """

    def __init__(self, *, backend: XLSTMBackend = "vanilla") -> None:
        super().__init__()
        if backend not in ("vanilla", "cuda"):
            raise ValueError("backend must be vanilla or cuda")
        self.backend = backend
        self.core_config = XLSTMMixerConfig()
        self.revin = NonAffineRevIN(5, epsilon=self.core_config.revin_epsilon)
        self.time_projection = nn.Linear(64, 1)
        self.up_projection = nn.Linear(1, 128)
        self.initial_token = nn.Parameter(torch.empty(1, 128))
        nn.init.normal_(self.initial_token, mean=0.0, std=0.01)
        self.slstm_stack = build_xlstm_stack(self.core_config, backend)
        self.output_projection = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (64, 5) or not x.is_floating_point():
            raise ValueError("XM-C8 contexts must be floating [B,64,5]")
        normalized, statistics = self.revin.normalize(x)
        last = normalized[:, -1:, :].detach()
        preliminary = self.time_projection((normalized - last).transpose(1, 2))
        preliminary = preliminary + last.transpose(1, 2)
        embedded = self.up_projection(preliminary)
        initial = self.initial_token.unsqueeze(0).expand(len(x), -1, -1)
        tokens = torch.cat((initial, embedded), dim=1)
        # Released FULL semantics and call order, not reversed variate order.
        reverse = self.slstm_stack(torch.flip(tokens, dims=(-1,)))
        forward = self.slstm_stack(tokens)
        if reverse.shape != tokens.shape or forward.shape != tokens.shape:
            raise RuntimeError("sLSTM stack changed the token shape")
        close_token = torch.cat((forward, reverse), dim=-1)[:, 1 + 3, :]
        normalized_close = self.output_projection(close_token)
        prediction = (
            normalized_close * statistics.standard_deviation[:, 0, 3:4]
            + statistics.mean[:, 0, 3:4]
        )
        if prediction.shape != (len(x), 1) or not torch.isfinite(prediction).all():
            raise FloatingPointError("invalid XM-C8 endpoint prediction")
        return prediction

    @staticmethod
    def endpoint_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 2 or prediction.shape[1] != 1 or target.shape != prediction.shape:
            raise ValueError("XM-C8 prediction and target must both be [B,1]")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("XM-C8 prediction and target must be finite")
        loss = F.mse_loss(prediction, target, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite XM-C8 MSE")
        return loss

    def get_extra_state(self) -> dict[str, Any]:
        architecture = architecture_manifest()
        return {
            "schema_version": MODEL_SCHEMA,
            "architecture": architecture,
            "architecture_sha256": hashlib.sha256(
                json.dumps(architecture, sort_keys=True).encode()
            ).hexdigest(),
            "source_contract": SOURCE_CONTRACT.to_dict(),
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("XM-C8 checkpoint architecture/source mismatch")
