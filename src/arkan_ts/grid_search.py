from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any, Optional
import itertools
import numpy as np
from torch.utils.data import DataLoader
from statsmodels.tsa.arima.model import ARIMA

from .memory import ARMemory, ARMemoryConfig
from .model import ARKAN
from .train import SupervisedTS, train_arkan


@dataclass(frozen=True)
class ARIMAConfig:
    p: int
    d: int
    q: int


@dataclass(frozen=True)
class ARKANConfig:
    p: int
    hidden: int
    n_basis: int
    basis: str = "rbf"  # "rbf" | "bspline"


def eval_split(series: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ARMemory]:
    """
    Build (Ztr, ytr, Zte, yte) with an AR memory fitted only on the training split.
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D.")
    N = series.shape[0]
    Ntr = int(0.8 * N)
    train = series[:Ntr]
    mem = ARMemory(ARMemoryConfig(p=p, standardize=True)).fit(train)
    Ztr = mem.transform(train)
    ytr = mem.target(train)

    Zall = mem.transform(series)
    Zte = Zall[Ntr - p:]           # align to test region
    yte = series[p:][Ntr - p:]
    return Ztr, ytr, Zte, yte, mem


def grid_search_arima(series: np.ndarray,
                      grid: Iterable[ARIMAConfig]) -> Tuple[ARIMAConfig, float, Dict[str, Any]]:
    """
    Grid-search ARIMA using recursive one-step forecasting on the last 20% of the series.
    Returns (best_cfg, best_mse, report).
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D.")
    N = series.shape[0]
    Ntr = int(0.8 * N)
    train, test = series[:Ntr], series[Ntr:]

    report: Dict[str, Any] = {}
    best: Tuple[Optional[ARIMAConfig], float] = (None, float("inf"))

    for cfg in grid:
        try:
            preds = []
            hist = list(train)
            for x_true in test:
                m = ARIMA(hist, order=(cfg.p, cfg.d, cfg.q)).fit(method_kwargs={"warn_convergence": False})
                pred = float(m.forecast(steps=1)[0])
                preds.append(pred)
                hist.append(x_true)  # recursive update with ground-truth
            mse = float(((np.array(preds) - test) ** 2).mean())
            report[f"ARIMA(p={cfg.p},d={cfg.d},q={cfg.q})"] = mse
            if mse < best[1]:
                best = (cfg, mse)
        except Exception as e:  # pragma: no cover - robust to convergence fails
            report[f"ARIMA(p={cfg.p},d={cfg.d},q={cfg.q})"] = f"error: {e}"

    if best[0] is None:
        raise RuntimeError("All ARIMA fits failed.")
    return best[0], best[1], report


def grid_search_arkan(series: np.ndarray,
                      grid: Iterable[ARKANConfig],
                      batch_size: int = 64,
                      epochs: int = 200,
                      lr: float = 3e-3,
                      device: str = "cpu") -> Tuple[ARKANConfig, float, Dict[str, Any]]:
    """
    Grid-search AR-KAN (memory p from cfg.p; varies hidden, n_basis, basis).
    Returns (best_cfg, best_mse, report).
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D.")
    report: Dict[str, Any] = {}
    best: Tuple[Optional[ARKANConfig], float] = (None, float("inf"))
    split_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ARMemory]] = {}

    for cfg in grid:
        if cfg.p not in split_cache:
            split_cache[cfg.p] = eval_split(series, p=cfg.p)
        Ztr, ytr, Zte, yte, _ = split_cache[cfg.p]

        ds_tr = SupervisedTS(Ztr, ytr)
        ds_te = SupervisedTS(Zte, yte)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        model = ARKAN(p=cfg.p, hidden=int(cfg.hidden), n_basis=int(cfg.n_basis), basis=cfg.basis)
        _, test_hist = train_arkan(model, dl_tr, dl_te, epochs=epochs, lr=lr, device=device)
        mse = float(test_hist[-1])
        key = f"ARKAN(p={cfg.p},hidden={cfg.hidden},n_basis={cfg.n_basis},basis={cfg.basis})"
        report[key] = mse

        if mse < best[1]:
            best = (cfg, mse)

    if best[0] is None:
        raise RuntimeError("All AR-KAN fits failed.")
    return best[0], best[1], report


def default_arima_grid() -> List[ARIMAConfig]:
    """Small but meaningful ARIMA search space."""
    return [ARIMAConfig(p, d, q) for p, d, q in itertools.product((10, 20, 30), (0, 1), (0, 1, 2))]


def default_arkan_grid(basis: str = "rbf") -> List[ARKANConfig]:
    """Small AR-KAN hyperparameter grid."""
    return [
        ARKANConfig(p, hidden, n_basis, basis=basis)
        for p, hidden, n_basis in itertools.product((10, 20, 30), (25, 50, 100), (4, 8, 12))
    ]
