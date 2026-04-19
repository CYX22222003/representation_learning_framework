from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceVAE(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        flat_dim = seq_len * input_dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.reshape(x.shape[0], -1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.reshape(z.shape[0], self.seq_len, self.input_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(
    recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kld
    return total, recon, kld


def train_vae_epoch(
    model: SequenceVAE,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    beta: float = 1.0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)

        optimizer.zero_grad()
        recon, mu, logvar, _ = model(x)
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
