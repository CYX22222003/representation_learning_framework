from file_list import one_day_file_list, four_hour_file_list, one_hour_file_list, DATA_DIR
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class MarketDataset(Dataset):
    def __init__(self, sequences):
        # sequences: [N, seq_len, features]
        self.data = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # unsupervised → return only input
        return self.data[idx]
    

def preprocess_market(df):
    df = df.ffill().interpolate()
    price_cols = ["open", "high", "low", "close"]
    volume_col = ["volume"]
    price = df[price_cols].values.astype(np.float32)
    volume = df[volume_col].values.astype(np.float32)
    volume_mean = volume.mean(axis=0)
    volume_std = volume.std(axis=0) + 1e-8
    volume = (volume - volume_mean) / volume_std
    features = np.concatenate([price, volume], axis=1)
    return features

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i:i + seq_len]
        sequences.append(seq)
    return np.array(sequences)

def split_sequences(sequences, train_ratio=0.8):
    split_idx = int(len(sequences) * train_ratio)
    return sequences[:split_idx], sequences[split_idx:]

def build_from_file_list(file_list, seq_len):
    train_all = []
    test_all = []

    for filename, _ in file_list:
        path = os.path.join(DATA_DIR, filename)

        try:
            df = pd.read_feather(path)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        if len(df) == 0:
            continue

        features = preprocess_market(df)
        sequences = create_sequences(features, seq_len)

        if len(sequences) == 0:
            continue

        train_seq, test_seq = split_sequences(sequences)

        train_all.append(train_seq)
        test_all.append(test_seq)
    if len(train_all) == 0:
        raise ValueError("No valid training data")

    train_all = np.concatenate(train_all, axis=0)
    test_all = np.concatenate(test_all, axis=0)

    return train_all, test_all

if __name__ == "__main__":
    SEQ_LEN = 64
    train_4h, test_4h = build_from_file_list(four_hour_file_list, SEQ_LEN)
    train_dataset = MarketDataset(train_4h)
    test_dataset = MarketDataset(test_4h)
    
    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    sample = train_dataset[0]
    print("Sample shape:", sample.shape)
    
    import matplotlib.pyplot as plt

    sample = train_dataset[0].numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(sample[:, 0], label="open")
    plt.plot(sample[:, 1], label="high")
    plt.plot(sample[:, 2], label="low")
    plt.plot(sample[:, 3], label="close")
    plt.legend()
    plt.title("Price sequence")

    plt.savefig("debug_sequence.png")
    plt.close()