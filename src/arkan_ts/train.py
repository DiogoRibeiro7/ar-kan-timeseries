from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SupervisedTS(Dataset):
    """Thin dataset wrapper for (features, targets)."""

    def __init__(self, X, y) -> None:
        import numpy as np

        if not isinstance(X, (np.ndarray, torch.Tensor)):
            raise TypeError("X must be a numpy array or torch tensor.")
        if not isinstance(y, (np.ndarray, torch.Tensor)):
            raise TypeError("y must be a numpy array or torch tensor.")

        X_np = X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()
        y_np = y if isinstance(y, np.ndarray) else y.detach().cpu().numpy()

        if X_np.ndim != 2:
            raise ValueError("X must be 2D (N, p).")
        if y_np.ndim != 1:
            raise ValueError("y must be 1D (N,).")
        if X_np.shape[0] != y_np.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        self.X = torch.from_numpy(X_np.astype("float32"))
        self.y = torch.from_numpy(y_np.astype("float32"))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def train_arkan(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 200,
    lr: float = 3e-3,
    device: str = "cpu",
) -> tuple[list[float], list[float]]:
    """Standard MSE training loop with Adam. Returns train/test MSE histories."""
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_hist: list[float] = []
    test_hist: list[float] = []

    for _ in range(int(epochs)):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
        tr_mse = tr_loss / len(train_loader.dataset)
        train_hist.append(tr_mse)

        # Eval
        model.eval()
        te_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = mse(pred, yb)
                te_loss += float(loss.item()) * xb.size(0)
        te_mse = te_loss / len(test_loader.dataset)
        test_hist.append(te_mse)

    return train_hist, test_hist
