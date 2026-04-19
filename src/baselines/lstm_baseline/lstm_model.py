import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from scripts.file_list import DATA_DIR, four_hour_file_list
import os
import pandas as pd

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.abs((y_true - y_pred) / (y_true + self.eps))
        return torch.mean(loss)

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
    
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, patience=50, device="cuda"):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MAPELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ---- TRAIN ----
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

        # ---- VALIDATE ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # ---- EARLY STOPPING ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model)
    return model

def evaluate(model, test_loader, device="cuda"):
    model.eval()
    criterion = MAPELoss()

    total_loss = 0

    preds = []
    targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            total_loss += loss.item()

            preds.append(pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)

    print("Test MAPE:", total_loss / len(test_loader))

    return preds, targets

if __name__ == "__main__":
    SEQ_LEN = 64

    X_train, y_train, X_test, y_test = build_lstm_dataset(
        four_hour_file_list, SEQ_LEN
    )

    train_dataset = PriceDataset(X_train, y_train)
    test_dataset = PriceDataset(X_test, y_test)

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    # sample check
    X_sample, y_sample = train_dataset[0]
    print("Input shape:", X_sample.shape)  # [64, 1]
    print("Target:", y_sample)
    
    from torch.utils.data import DataLoader, random_split

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"devise used: {device}")
    
    model = LSTMModel()

    model = train_model(
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=1e-3,
        patience=10,
        device=device
    )

    preds, targets = evaluate(model, test_loader, device=device)
    model_path = "./event_stacked_lstm.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")