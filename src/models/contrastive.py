from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    """
    CNN backbone + projector head for contrastive pretraining.

    The backbone uses two Conv1d layers with an adaptive average pool to produce
    a fixed-size representation regardless of sequence length.  The projector
    maps the backbone output to a normalised embedding used only during training
    (NT-Xent loss).  After pretraining, the backbone hidden state `h` (not `z`)
    is typically used as the frozen embedding for downstream tasks.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            # Conv1d operates on [B, channels, time]; input_dim is the channel axis.
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # Pool to length 1 so output is [B, hidden_dim, 1] regardless of seq_len.
            nn.AdaptiveAvgPool1d(1),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F] → transpose to [B, F, T] for Conv1d
        h = self.backbone(x.transpose(1, 2)).squeeze(-1)  # [B, hidden_dim]
        z = F.normalize(self.projector(h), dim=-1)        # [B, embedding_dim], unit norm
        return h, z


# ── Data augmentations ─────────────────────────────────────────────────────────
# Each augmentation creates a different "view" of the same sequence.
# The model learns that both views should map to similar embeddings.

def jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Add small Gaussian noise to each timestep."""
    return x + torch.randn_like(x) * sigma


def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Multiply the sequence by a random scalar close to 1 (applied per sample)."""
    scale = 1.0 + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * sigma
    return x * scale


def time_mask(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
    """Zero out a contiguous segment of timesteps (different start per sample)."""
    out = x.clone()
    t = x.shape[1]
    mask_len = max(1, int(t * mask_ratio))
    start = torch.randint(0, max(1, t - mask_len + 1), (x.shape[0],), device=x.device)
    for i in range(x.shape[0]):
        out[i, start[i] : start[i] + mask_len] = 0.0
    return out


def make_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two augmented views of the same batch.
    View 1: scale → jitter.  View 2: time_mask → scale.
    Different augmentation chains make the task harder and richer.
    """
    v1 = scaling(jitter(x))
    v2 = time_mask(scaling(x))
    return v1, v2


# ── NT-Xent loss ──────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Normalised temperature-scaled cross-entropy (SimCLR).

    Positive pairs: (z1[i], z2[i]) — two views of the same sequence.
    Negative pairs: all other combinations within the batch.
    The model is trained to make positive pairs more similar than negatives.
    """
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have identical shapes")
    batch_size = z1.shape[0]
    # Stack both views: [2B, dim]; similarities are computed for all pairs.
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    # Mask out self-similarities (diagonal) so a sample isn't its own positive.
    logits_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(~logits_mask, float("-inf"))

    # For each view-1 sample i its positive is view-2 sample i (offset by batch_size).
    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(sim, targets)
