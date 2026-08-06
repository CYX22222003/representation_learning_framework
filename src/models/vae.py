from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceVAE(nn.Module):
    """
    Variational Autoencoder for fixed-length OHLCV sequences.

    The encoder maps a flattened [seq_len * input_dim] sequence to a latent
    distribution (mu, logvar).  The decoder reconstructs the original sequence
    from a sampled latent vector.  After pretraining, only the encoder is used
    (frozen) to extract the latent mean `mu` as the embedding for each sequence.
    """

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
        # Two separate heads produce the mean and log-variance of the latent distribution.
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
        # Reparameterization trick: z = mu + eps * std lets gradients flow through
        # the sampling step during backprop (eps is a fixed noise sample).
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
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Reconstruction loss: how well the decoder reproduces the input sequence.
    recon = F.mse_loss(recon_x, x, reduction="mean")
    # KL divergence: regularises the latent space toward a unit Gaussian.
    # beta > 1 (β-VAE) increases disentanglement at the cost of reconstruction quality.
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + beta * kld
    return total, recon, kld