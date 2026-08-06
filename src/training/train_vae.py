from __future__ import annotations

import torch

from models.vae import SequenceVAE, vae_loss


def train_vae_epoch(
    model: SequenceVAE,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    beta: float = 1.0,
) -> dict[str, float]:
    """
    Run one full epoch of VAE pretraining (unsupervised).

    Returns averaged loss components over all batches:
        loss  — total β-VAE loss (recon + beta * KLD)
        recon — reconstruction MSE
        kld   — KL divergence from unit Gaussian prior
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    n_batches = 0

    for batch in dataloader:
        # Dataloader may return (x,) tuples or bare tensors depending on Dataset.
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        optimizer.zero_grad()
        recon, mu, logvar, _ = model(x)  # z (last return) unused during training
        loss, recon_loss, kld = vae_loss(recon, x, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(recon_loss.item())
        total_kld += float(kld.item())
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Empty dataloader")

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kld": total_kld / n_batches,
    }
