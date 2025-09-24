from __future__ import annotations

import argparse
import math
import numpy as np
import torch

from .grid_search import default_arima_grid, default_arkan_grid, grid_search_arima, grid_search_arkan
from .memory import ARMemory, ARMemoryConfig
from .model import ARKAN
from .train import SupervisedTS, train_arkan


def _make_f1(t: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.cos(2.0 * t) + np.cos(2.0 * math.pi * t) + rng.normal(0.0, sigma, size=t.shape[0])


def _make_f2(t: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.sin(3.0 * t) + np.sin(2.0 * math.e * t) + rng.normal(0.0, sigma, size=t.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="AR-KAN TS utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="Run synthetic f1/f2 demo")
    demo.add_argument("--sigma", type=float, default=0.2)
    demo.add_argument("--p", type=int, default=20)
    demo.add_argument("--epochs", type=int, default=300)
    demo.add_argument("--basis", choices=["rbf", "bspline"], default="rbf")
    demo.add_argument("--device", default="cpu")

    grid = sub.add_parser("grid", help="Run grid-search over synthetic f1")
    grid.add_argument("--basis", choices=["rbf", "bspline"], default="bspline")
    grid.add_argument("--epochs", type=int, default=200)
    grid.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.cmd == "demo":
        rng = np.random.default_rng(0)
        N = 500
        t = np.linspace(0.0, 8.0 * math.pi, N, endpoint=True)
        x = _make_f1(t, sigma=float(args.sigma), rng=rng)

        # Split 80/20 and train/eval AR-KAN with chosen basis
        Ntr = int(0.8 * N)
        mem = ARMemory(ARMemoryConfig(p=int(args.p), standardize=True)).fit(x[:Ntr])
        Ztr, ytr = mem.transform(x[:Ntr]), mem.target(x[:Ntr])
        Zte, yte = mem.transform(x)[Ntr - args.p:], x[args.p:][Ntr - args.p:]

        ds_tr = SupervisedTS(Ztr, ytr)
        ds_te = SupervisedTS(Zte, yte)

        model = ARKAN(p=int(args.p), hidden=50, n_basis=8, basis=args.basis)
        _, test_hist = train_arkan(
            model,
            torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True),
            torch.utils.data.DataLoader(ds_te, batch_size=64, shuffle=False),
            epochs=int(args.epochs),
            lr=3e-3,
            device=args.device,
        )
        print(f"AR-KAN ({args.basis}) Test MSE: {test_hist[-1]:.6f}")

    elif args.cmd == "grid":
        rng = np.random.default_rng(0)
        N = 500
        t = np.linspace(0.0, 8.0 * math.pi, N, endpoint=True)
        x = _make_f1(t, sigma=0.2, rng=rng)

        best_a, mse_a, _ = grid_search_arima(x, default_arima_grid())
        best_k, mse_k, _ = grid_search_arkan(x, default_arkan_grid(basis=args.basis),
                                             epochs=int(args.epochs), device=args.device)
        print("Best ARIMA:", best_a, "MSE=", mse_a)
        print(f"Best AR-KAN ({args.basis}):", best_k, "MSE=", mse_k)


if __name__ == "__main__":
    main()
