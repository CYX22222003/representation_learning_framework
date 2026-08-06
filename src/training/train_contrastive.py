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
    """
    Run one full epoch of contrastive pretraining (unsupervised).

    For each batch: two augmented views are created from the same sequences,
    both are encoded, and NT-Xent loss pushes the two views of the same
    sequence together while pushing different sequences apart.

    Returns the average NT-Xent loss over all batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        # Create two differently augmented views of the same batch.
        v1, v2 = make_views(x)

        optimizer.zero_grad()
        # h (backbone output) is unused during training; only projected z enters the loss.
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
