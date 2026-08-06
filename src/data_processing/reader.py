from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def read_market_feather(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_feather(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def load_processed_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    if "train" not in data or "test" not in data:
        raise ValueError("Processed .npz must contain 'train' and 'test'")
    return data["train"].astype(np.float32), data["test"].astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray) -> None:
        self.x = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


def build_sequence_dataloader(
    npz_path: str, split: str = "train", batch_size: int = 64, shuffle: bool | None = None
) -> DataLoader:
    train, test = load_processed_npz(npz_path)
    if split not in {"train", "test"}:
        raise ValueError("split must be one of {'train', 'test'}")

    arr = train if split == "train" else test
    dataset = SequenceDataset(arr)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
