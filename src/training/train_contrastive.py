from __future__ import annotations

import torch

from models.contrastive import ContrastiveEncoder, make_views, nt_xent_loss


def train_contrastive_epoch(
    model: ContrastiveEncoder,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    temperature: float = 0.2,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        v1, v2 = make_views(x)

        optimizer.zero_grad()
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Empty dataloader")
    return total_loss / n_batches
