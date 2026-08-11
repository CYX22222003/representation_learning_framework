from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceCNNBackbone(nn.Module):
    """
    CNN sequence backbone shared by BYOL's online and target encoders.

    The architecture intentionally matches the existing contrastive encoder
    backbone so BYOL-vs-SimCLR comparisons isolate the pretraining objective
    instead of changing encoder capacity.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(-1)


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class BYOLEncoder(nn.Module):
    """
    Bootstrap Your Own Latent encoder for fixed-length OHLCV sequences.

    BYOL trains an online encoder/projector/predictor to predict the target
    encoder's projected representation of another augmented view. The target
    encoder/projector are updated by exponential moving average and do not
    receive gradients. After pretraining, the online backbone hidden state `h`
    is used as the frozen downstream embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        predictor_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.predictor_hidden_dim = predictor_hidden_dim

        self.online_backbone = SequenceCNNBackbone(input_dim=input_dim, hidden_dim=hidden_dim)
        self.online_projector = _make_mlp(hidden_dim, hidden_dim, projection_dim)
        self.online_predictor = _make_mlp(projection_dim, predictor_hidden_dim, projection_dim)

        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        self._freeze_target()

    def _freeze_target(self) -> None:
        for module in (self.target_backbone, self.target_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward_online(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.online_backbone(x)
        z = F.normalize(self.online_projector(h), dim=-1)
        p = self.online_predictor(z)
        return h, z, p

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        h = self.target_backbone(x)
        return F.normalize(self.target_projector(h), dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the online backbone representation used downstream."""
        return self.online_backbone(x)

    @torch.no_grad()
    def update_target(self, tau: float = 0.99) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be in [0, 1]")
        online_modules = (self.online_backbone, self.online_projector)
        target_modules = (self.target_backbone, self.target_projector)
        for online_module, target_module in zip(online_modules, target_modules):
            for online_param, target_param in zip(
                online_module.parameters(), target_module.parameters()
            ):
                target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h1, _, p1 = self.forward_online(view1)
        h2, _, p2 = self.forward_online(view2)
        z1_target = self.forward_target(view1)
        z2_target = self.forward_target(view2)
        return h1, h2, p1, p2, z1_target, z2_target


def byol_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Negative cosine regression loss used by BYOL.

    The target is detached so gradients flow only through the online network.
    Range is approximately [0, 4], with 0 meaning identical directions.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (prediction * target).sum(dim=-1).mean()


def byol_loss(
    p1: torch.Tensor,
    z2_target: torch.Tensor,
    p2: torch.Tensor,
    z1_target: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        byol_prediction_loss(p1, z2_target)
        + byol_prediction_loss(p2, z1_target)
    )
