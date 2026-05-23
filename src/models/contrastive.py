from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F] -> [B, F, T]
        h = self.backbone(x.transpose(1, 2)).squeeze(-1)
        z = F.normalize(self.projector(h), dim=-1)
        return h, z


def jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    scale = 1.0 + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * sigma
    return x * scale


def time_mask(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
    out = x.clone()
    t = x.shape[1]
    mask_len = max(1, int(t * mask_ratio))
    start = torch.randint(0, max(1, t - mask_len + 1), (x.shape[0],), device=x.device)
    for i in range(x.shape[0]):
        out[i, start[i] : start[i] + mask_len] = 0.0
    return out


def make_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v1 = scaling(jitter(x))
    v2 = time_mask(scaling(x))
    return v1, v2


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have identical shapes")
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    logits_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~logits_mask, float("-inf"))

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(sim, targets)
