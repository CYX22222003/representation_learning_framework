from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines.ta_mlp_baseline.ta_features import N_FEATURES, build_ta_dataset


CLASS_NAMES_TRICLASS = ("BUY", "HOLD", "SELL")
CLASS_NAMES_BINARY = ("DOWN", "UP")


class TAMLPClassifier(nn.Module):
    """
    MLP classifier trained on TA-Lib technical indicator features.

    Architecture from the FreqTrade-based MLP paper:
    in_dim → 128 → 64 → 32 → n_classes, LeakyReLU activations.

    Default input dimension matches the 36-feature vector produced by
    ``compute_ta_features()`` in ``ta_features.py``.

    Parameters
    ----------
    in_dim : int
        Number of input features (default: N_FEATURES = 36).
    n_classes : int
        Number of output classes.  Use 2 for binary trend classification
        (up / down), 3 for the original BUY / HOLD / SELL formulation.
    """

    def __init__(self, in_dim: int = N_FEATURES, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, in_dim]

        Returns
        -------
        logits : Tensor [B, n_classes]
        """
        return self.net(x)


def train_model(model, train_loader, epochs=30, lr=1e-3, device="cuda"):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch}: train={train_loss:.4f}")
        history["train_loss"].append(train_loss)

    return model, history


def evaluate(model, test_loader, n_classes, device="cuda"):
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = logits.argmax(dim=1)

    accuracy = (preds == targets).float().mean().item()

    # Confusion matrix (rows=actual, cols=predicted).
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t, p] += 1

    # Per-class precision / recall / F1.
    per_class = []
    for c in range(n_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append((prec, rec, f1))

    macro_f1 = float(np.mean([f1 for _, _, f1 in per_class]))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "logits": logits,
        "preds": preds,
        "targets": targets,
    }


def _print_eval(results, class_names):
    print(f"Test accuracy : {results['accuracy']:.4f}")
    print(f"Test macro-F1 : {results['macro_f1']:.4f}")
    print("Per-class metrics:")
    print(f"  {'class':<6} {'precision':>10} {'recall':>10} {'f1':>10}")
    for name, (p, r, f) in zip(class_names, results["per_class"]):
        print(f"  {name:<6} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = "         " + " ".join(f"{c:>8}" for c in class_names)
    print(header)
    cm = results["confusion_matrix"]
    for i, name in enumerate(class_names):
        row = " ".join(f"{cm[i, j].item():>8d}" for j in range(len(class_names)))
        print(f"  {name:<6} {row}")


if __name__ == "__main__":
    import argparse
    import random
    from datetime import date

    from data_processing.file_list import DATA_DIR, four_hour_file_list

    parser = argparse.ArgumentParser(description="Train the TA-MLP baseline.")
    parser.add_argument(
        "--run-name",
        default=f"{date.today().isoformat()}-default",
        help="Subdirectory under experiments/ where artifacts are written.",
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (no early stopping; "
                             "matches framework convention of no val split).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for torch / numpy / random / CUDA.")
    parser.add_argument("--label-mode", choices=("triclass", "binary"),
                        default="triclass",
                        help="triclass = BUY/HOLD/SELL (upstream paper formulation), "
                             "binary = up/down over --horizon.")
    parser.add_argument("--b-window", type=int, default=5,
                        help="Backward EWM span for triclass label MA.")
    parser.add_argument("--f-window", type=int, default=2,
                        help="Forward look-ahead for triclass label.")
    parser.add_argument("--hold-q", type=float, default=0.85,
                        help="Quantile of |pct_change| for triclass alpha (HOLD threshold).")
    parser.add_argument("--buy-sell-q", type=float, default=0.997,
                        help="Quantile of |pct_change| for triclass beta cap.")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forward horizon for binary label (binary mode only).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(baseline_dir, "experiments", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Config: epochs={args.epochs}, seed={args.seed}, label_mode={args.label_mode}")
    if args.label_mode == "triclass":
        print(f"  b_window={args.b_window}, f_window={args.f_window}, "
              f"hold_q={args.hold_q}, buy_sell_q={args.buy_sell_q}")
        n_classes = 3
        class_names = CLASS_NAMES_TRICLASS
    else:
        print(f"  horizon={args.horizon}")
        n_classes = 2
        class_names = CLASS_NAMES_BINARY

    feather_paths = [os.path.join(DATA_DIR, fn) for fn, _ in four_hour_file_list]
    X_train, y_train, X_test, y_test = build_ta_dataset(
        feather_paths,
        horizon=args.horizon,
        label_mode=args.label_mode,
        b_window=args.b_window,
        f_window=args.f_window,
        hold_q=args.hold_q,
        buy_sell_q=args.buy_sell_q,
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size : {len(X_test)}")
    print(f"Input dim : {X_train.shape[1]}")
    train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
    test_dist = dict(zip(*np.unique(y_test, return_counts=True)))
    print(f"Train class dist: {train_dist}")
    print(f"Test  class dist: {test_dist}")

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train_t)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test_t)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = TAMLPClassifier(in_dim=X_train.shape[1], n_classes=n_classes)
    model, history = train_model(
        model, train_loader,
        epochs=args.epochs, lr=1e-3, device=device,
    )

    results = evaluate(model, test_loader, n_classes=n_classes, device=device)
    _print_eval(results, class_names)

    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    history_path = os.path.join(run_dir, "history.npz")
    preds_path = os.path.join(run_dir, "predictions.npz")
    cm_path = os.path.join(run_dir, "confusion_matrix.npz")

    torch.save(model.state_dict(), checkpoint_path)
    np.savez(
        history_path,
        train_loss=np.array(history["train_loss"], dtype=np.float32),
        epochs=np.int32(args.epochs),
        seed=np.int32(args.seed),
    )
    np.savez(
        preds_path,
        preds=results["preds"].numpy().astype(np.int64),
        logits=results["logits"].numpy().astype(np.float32),
        targets=results["targets"].numpy().astype(np.int64),
    )
    np.savez(
        cm_path,
        cm=results["confusion_matrix"].numpy().astype(np.int64),
        class_names=np.array(class_names),
    )

    print(f"Checkpoint        saved to {checkpoint_path}")
    print(f"History           saved to {history_path}")
    print(f"Predictions       saved to {preds_path}")
    print(f"Confusion matrix  saved to {cm_path}")
