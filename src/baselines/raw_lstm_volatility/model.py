from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class VolatilitySequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, row_indices: np.ndarray) -> None:
        arr = np.asarray(sequences, dtype=np.float32)
        y = np.asarray(labels, dtype=np.float32).reshape(-1)
        idx = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if arr.ndim != 3:
            raise ValueError(f"Expected sequences [N, seq_len, features], got {arr.shape}")
        if y.shape[0] != idx.shape[0]:
            raise ValueError("labels and row_indices must have the same length")
        if idx.shape[0] and (idx.min() < 0 or idx.max() >= arr.shape[0]):
            raise ValueError("row_indices are out of range for sequences")
        self.sequences = torch.tensor(arr[idx], dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.row_indices = idx.copy()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


class RawLSTMVolatility(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: Dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    criterion = nn.MSELoss()
    model.train()
    total = 0.0
    count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch = int(x_batch.shape[0])
        total += float(loss.item()) * batch
        count += batch
    return total / max(count, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x_batch, y_batch in loader:
        preds.append(model(x_batch.to(device)).cpu())
        targets.append(y_batch.cpu())
    return (
        torch.cat(preds, dim=0).reshape(-1).numpy().astype(np.float32),
        torch.cat(targets, dim=0).reshape(-1).numpy().astype(np.float32),
    )
