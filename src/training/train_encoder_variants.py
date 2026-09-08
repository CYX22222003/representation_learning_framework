from __future__ import annotations

import math

import torch

from models.contrastive import make_views, nt_xent_loss
from models.encoder_variants import TemporalContrastiveEncoder


def train_temporal_contrastive_epoch(
    model: TemporalContrastiveEncoder,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    temperature: float = 0.2,
) -> dict[str, float]:
    """Train one NT-Xent epoch and return representation health diagnostics."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    model.train()
    totals = {"loss": 0.0, "embedding_std": 0.0, "embedding_norm": 0.0, "projected_std": 0.0}
    n_batches = 0

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        view1, view2 = make_views(x)

        optimizer.zero_grad()
        h1, z1 = model(view1)
        h2, z2 = model(view2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite NT-Xent loss")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("non-finite gradient in contrastive encoder")
        optimizer.step()

        with torch.no_grad():
            hidden = torch.cat([h1, h2], dim=0)
            projected = torch.cat([z1, z2], dim=0)
            diagnostics = {
                "loss": loss,
                "embedding_std": hidden.std(dim=0, unbiased=False).mean(),
                "embedding_norm": hidden.norm(dim=-1).mean(),
                "projected_std": projected.std(dim=0, unbiased=False).mean(),
            }
        for name, value in diagnostics.items():
            scalar = float(value.item())
            if not math.isfinite(scalar):
                raise FloatingPointError(f"non-finite {name} diagnostic")
            totals[name] += scalar
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Empty dataloader")
    return {name: total / n_batches for name, total in totals.items()}
