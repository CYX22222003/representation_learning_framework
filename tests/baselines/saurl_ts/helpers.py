from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from baselines.saurl_ts import SaURLConfig


def small_config() -> SaURLConfig:
    return SaURLConfig(
        sequence_length=8,
        input_dim=5,
        augmentation_hidden_dim=4,
        encoder_hidden_dim=8,
        representation_dim=16,
        projector_hidden_dim=16,
        projection_dim=16,
        predictor_hidden_dim=16,
        dilations=(1, 2),
        attention_regions=4,
        dropout=0.0,
    )


def set_trainable(parameters: Iterable[nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)
        if not enabled:
            parameter.grad = None


def clone_parameters(parameters: Iterable[nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def any_parameter_changed(
    before: list[torch.Tensor], parameters: Iterable[nn.Parameter]
) -> bool:
    return any(
        not torch.equal(old, new.detach()) for old, new in zip(before, parameters)
    )
