from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from arkan_ts.memory import ARMemory, ARMemoryConfig
from arkan_ts.model import ARKAN
from arkan_ts.train import SupervisedTS, train_arkan


def make_f1(t: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.cos(2.0 * t) + np.cos(2.0 * math.pi * t) + rng.normal(0.0, sigma, size=t.shape[0])


def make_f2(t: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.sin(3.0 * t) + np.sin(2.0 * math.e * t) + rng.normal(0.0, sigma, size=t.shape[0])


def run_one(series: np.ndarray, p: int = 20, epochs: int = 300, basis: str = "rbf") -> float:
    """Train AR-KAN on a series and return test MSE."""
    N = len(series)
    Ntr = int(0.8 * N)
    train = series[:Ntr]

    mem = ARMemory(ARMemoryConfig(p=p, standardize=True)).fit(train)
    Ztr, ytr = mem.transform(train), mem.target(train)

    Zall = mem.transform(series)
    Zte, yte = Zall[Ntr - p:], series[p:][Ntr - p:]

    ds_tr = SupervisedTS(Ztr, ytr)
    ds_te = SupervisedTS(Zte, yte)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False)

    model = ARKAN(p=p, hidden=50, n_basis=8, basis=basis)
    _, test_hist = train_arkan(model, dl_tr, dl_te, epochs=epochs, lr=3e-3, device="cpu")
    return float(test_hist[-1])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 500
    t = np.linspace(0.0, 8.0 * math.pi, N, endpoint=True)

    for sigma in (0.1, 0.2, 0.3, 0.4):
        for name, maker in [("f1", make_f1), ("f2", make_f2)]:
            x = maker(t, sigma=sigma, rng=rng)
            mse = run_one(x, p=20, epochs=300, basis="bspline")
            print(f"{name}, σ={sigma:.1f} | AR-KAN(bspline) MSE={mse:.6f}")
