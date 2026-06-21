import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from data_processing.file_list import DATA_DIR, four_hour_file_list
import os
import pandas as pd

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2) + self.eps)

def create_labeled_sequences(close_prices, seq_len):
    X, y = [], []

    for i in range(len(close_prices) - seq_len):
        X.append(close_prices[i:i+seq_len])
        y.append(close_prices[i+seq_len])

    X = np.array(X)
    y = np.array(y)

    return X, y

def build_lstm_dataset(file_list, seq_len):
    X_all = []
    y_all = []

    for filename, _ in file_list:
        path = os.path.join(DATA_DIR, filename)

        try:
            df = pd.read_feather(path)
        except:
            continue

        df = df.ffill().interpolate()

        close = df["close"].values.astype(np.float32)

        if len(close) <= seq_len:
            continue

        X, y = create_labeled_sequences(close, seq_len)

        # split per market (chronological)
        split_idx = int(len(X) * 0.8)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_all.append((X_train, X_test))
        y_all.append((y_train, y_test))

    # merge across markets
    X_train = np.concatenate([x[0] for x in X_all], axis=0)
    X_test  = np.concatenate([x[1] for x in X_all], axis=0)

    y_train = np.concatenate([y[0] for y in y_all], axis=0)
    y_test  = np.concatenate([y[1] for y in y_all], axis=0)

    return X_train, y_train, X_test, y_test

class PriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(input_size=50, hidden_size=30, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)

        self.lstm3 = nn.LSTM(input_size=30, hidden_size=20, batch_first=True)
        self.dropout3 = nn.Dropout(0.05)

        self.fc = nn.Linear(20, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)
        x = x[:, -1, :]

        x = self.dropout3(x)
        x = self.fc(x)

        return x
    
def train_model(model, train_loader, epochs=30, lr=1e-3, device="cuda"):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = RMSELoss()

    history = {"train_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        print(f"Epoch {epoch}: train={train_loss:.4f}")
        history["train_loss"].append(train_loss)

    return model, history

def evaluate(model, test_loader, device="cuda"):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            preds.append(pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    print(f"Test MAE : {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")

    return preds, targets

if __name__ == "__main__":
    import argparse
    import random
    from datetime import date
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Train the LSTM baseline.")
    parser.add_argument(
        "--run-name",
        default=f"{date.today().isoformat()}-default",
        help="Subdirectory under experiments/ where artifacts are written.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (no early stopping; matches framework convention of no val split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch / numpy / random / CUDA. Makes runs reproducible and comparable across epoch budgets.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    SEQ_LEN = 64

    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(baseline_dir, "experiments", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Config: epochs={args.epochs}, seed={args.seed}")

    X_train, y_train, X_test, y_test = build_lstm_dataset(
        four_hour_file_list, SEQ_LEN
    )

    train_dataset = PriceDataset(X_train, y_train)
    test_dataset = PriceDataset(X_test, y_test)

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    X_sample, y_sample = train_dataset[0]
    print("Input shape:", X_sample.shape)
    print("Target:", y_sample)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = LSTMModel()
    model, history = train_model(
        model, train_loader,
        epochs=args.epochs, lr=1e-3, device=device,
    )

    preds, targets = evaluate(model, test_loader, device=device)

    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    history_path = os.path.join(run_dir, "history.npz")
    preds_path = os.path.join(run_dir, "predictions.npz")

    torch.save(model.state_dict(), checkpoint_path)
    np.savez(
        history_path,
        train_loss=np.array(history["train_loss"], dtype=np.float32),
        epochs=np.int32(args.epochs),
        seed=np.int32(args.seed),
    )
    np.savez(
        preds_path,
        preds=preds.numpy().reshape(-1).astype(np.float32),
        targets=targets.numpy().reshape(-1).astype(np.float32),
    )

    print(f"Checkpoint  saved to {checkpoint_path}")
    print(f"History     saved to {history_path}")
    print(f"Predictions saved to {preds_path}")