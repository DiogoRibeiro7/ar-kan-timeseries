from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


def _check_array_1d(x: np.ndarray, name: str) -> None:
    """Validate that x is a finite 1D float array."""
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray.")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError(f"{name} must be numeric.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values.")


def autocovariance(x: np.ndarray, lag: int) -> float:
    """
    Unbiased sample autocovariance at a given nonnegative lag.

    Parameters
    ----------
    x : np.ndarray
        1D numeric series.
    lag : int
        Nonnegative lag.

    Returns
    -------
    float
        Unbiased autocovariance at `lag`.
    """
    _check_array_1d(x, "x")
    if lag < 0:
        raise ValueError("lag must be nonnegative.")
    n = x.shape[0]
    x_centered = x - x.mean()
    if lag == 0:
        return float((x_centered @ x_centered) / (n - 1))
    return float(np.dot(x_centered[lag:], x_centered[:-lag]) / (n - lag - 1))


def yule_walker_ar(x: np.ndarray, p: int) -> Tuple[np.ndarray, float]:
    """
    Estimate AR(p) coefficients via Yule–Walker using Levinson–Durbin recursion.

    We use the sign convention: x_t ≈ sum_{i=1..p} a[i-1] * x_{t-i}

    Parameters
    ----------
    x : np.ndarray
        1D series (preferably standardized).
    p : int
        AR order (1 <= p < len(x)/5 recommended).

    Returns
    -------
    a : np.ndarray
        Coefficients a[0..p-1].
    noise_var : float
        Final prediction error variance from the recursion.

    Raises
    ------
    ValueError, RuntimeError
        If inputs are invalid or recursion fails.
    """
    _check_array_1d(x, "x")
    if not (1 <= p < x.shape[0] // 5):
        raise ValueError("Choose 1 <= p < len(x)/5 for stability.")
    gam = np.array([autocovariance(x, k) for k in range(p + 1)], dtype=float)
    if gam[0] <= 0.0:
        raise ValueError("Nonpositive variance; cannot fit AR.")

    a = np.zeros(p, dtype=float)
    e = gam[0]
    for k in range(1, p + 1):
        if k == 1:
            lam = gam[1] / e
            a[0] = lam
            e = e * (1.0 - lam**2)
            continue
        acc = gam[k] - np.dot(a[: k - 1], gam[1:k][::-1])
        lam = acc / e
        a_new = a.copy()
        a_new[: k - 1] = a[: k - 1] - lam * a[: k - 1][::-1]
        a_new[k - 1] = lam
        a = a_new
        e = e * (1.0 - lam**2)
        if e <= 0:
            raise RuntimeError("Levinson–Durbin failed: nonpositive error.")
    return a, float(e)


@dataclass
class ARMemoryConfig:
    """Configuration for the ARMemory module."""

    p: int
    standardize: bool = True


class ARMemory:
    """
    Data-driven AR(p) memory: pretrains coefficients then emits AR-weighted lags.

    For a 1D series x and fitted coeffs a[0..p-1], the feature at time t is:
        z_t[i] = a[i] * x_{t-i}

    If `standardize=True`, we standardize (mean/std) using the training split,
    and re-use those stats at transform time.
    """

    def __init__(self, cfg: ARMemoryConfig):
        if not isinstance(cfg.p, int) or cfg.p <= 0:
            raise ValueError("AR order p must be a positive integer.")
        self.cfg = cfg
        self._mean: Optional[float] = None
        self._std: Optional[float] = None
        self._a: Optional[np.ndarray] = None

    @property
    def coefficients(self) -> np.ndarray:
        """Return fitted AR coefficients, raising if not fitted."""
        if self._a is None:
            raise RuntimeError("ARMemory not fitted yet.")
        return self._a

    def fit(self, x: np.ndarray) -> "ARMemory":
        """Fit AR(p) to `x` and cache standardization stats if enabled."""
        _check_array_1d(x, "x")
        x_ = x.astype(float).copy()
        if self.cfg.standardize:
            self._mean = float(x_.mean())
            self._std = float(x_.std(ddof=1))
            if self._std == 0:
                raise ValueError("Series has zero variance.")
            x_ = (x_ - self._mean) / self._std
        self._a, _ = yule_walker_ar(x_, self.cfg.p)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Build AR-weighted lag matrix Z of shape (T - p, p), aligned for 1-step ahead.

        Z[t, i] = a[i] * x_{t + p - 1 - i},   with t = 0..T-p-1
        """
        if self._a is None:
            raise RuntimeError("fit must be called before transform.")
        _check_array_1d(x, "x")
        x_ = x.astype(float)
        if self.cfg.standardize:
            if self._mean is None or self._std is None:
                raise RuntimeError("Standardization params missing.")
            x_ = (x_ - self._mean) / self._std
        T = x_.shape[0]
        p = self.cfg.p
        if T <= p:
            raise ValueError("Series too short for the chosen AR order.")
        L = np.zeros((T - p, p), dtype=float)
        for i in range(p):
            L[:, i] = x_[p - 1 - i : T - i]
        Z = L * self.coefficients[None, :]
        return Z

    def target(self, x: np.ndarray) -> np.ndarray:
        """Targets aligned with transform(...): y[t] = x[p + t]."""
        _check_array_1d(x, "x")
        T = x.shape[0]
        p = self.cfg.p
        if T <= p:
            raise ValueError("Series too short for the chosen AR order.")
        return x[p:]
