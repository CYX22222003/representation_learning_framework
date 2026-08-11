from __future__ import annotations

import torch
import torch.nn.functional as F

from models.byol import BYOLEncoder, byol_loss
from models.contrastive import make_views


def train_byol_epoch(
    model: BYOLEncoder,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    target_decay: float = 0.99,
) -> dict[str, float]:
    """
    Run one full epoch of BYOL pretraining (unsupervised).

    Returns averaged diagnostics:
        loss          — symmetric BYOL prediction loss
        view_cosine   — cosine similarity of online predictions and target projections
        embedding_std — mean per-dimension std of online backbone embeddings
        embedding_norm — mean L2 norm of online backbone embeddings
    """
    model.train()
    total_loss = 0.0
    total_view_cosine = 0.0
    total_embedding_std = 0.0
    total_embedding_norm = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        v1, v2 = make_views(x)

        optimizer.zero_grad()
        h1, h2, p1, p2, z1_target, z2_target = model(v1, v2)
        loss = byol_loss(p1, z2_target, p2, z1_target)
        loss.backward()
        optimizer.step()
        model.update_target(tau=target_decay)

        with torch.no_grad():
            p1_norm = F.normalize(p1, dim=-1)
            p2_norm = F.normalize(p2, dim=-1)
            view_cosine = 0.5 * (
                F.cosine_similarity(p1_norm, z2_target, dim=-1).mean()
                + F.cosine_similarity(p2_norm, z1_target, dim=-1).mean()
            )
            h = torch.cat([h1, h2], dim=0)
            embedding_std = h.std(dim=0, unbiased=False).mean()
            embedding_norm = h.norm(dim=-1).mean()

        total_loss += float(loss.item())
        total_view_cosine += float(view_cosine.item())
        total_embedding_std += float(embedding_std.item())
        total_embedding_norm += float(embedding_norm.item())
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Empty dataloader")

    return {
        "loss": total_loss / n_batches,
        "view_cosine": total_view_cosine / n_batches,
        "embedding_std": total_embedding_std / n_batches,
        "embedding_norm": total_embedding_norm / n_batches,
    }
