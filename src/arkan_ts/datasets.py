from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

# Rdatasets CSV base URL. Note: requires internet to fetch datasets.
BASE = "https://vincentarelbundock.github.io/Rdatasets/csv"


@dataclass
class TimeSeriesWithMeta:
    """Container for a standardized univariate series and metadata."""

    name: str
    values: np.ndarray
    frequency: int | None
    periodicity_strength: float
    meta: dict[str, Any]


def _infer_period_via_acf(x: np.ndarray, nlags: int = 400) -> int | None:
    """Choose a seasonal period as the lag of the largest ACF peak (lag > 0).
    Returns None if no meaningful peak is found.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if len(x) < 10:
        return None
    ac = acf(x, nlags=min(nlags, len(x) - 2), fft=True)
    lags = np.arange(1, len(ac))
    vals = ac[1:]
    # Require reasonably positive correlation to count as "seasonal"
    k = int(lags[np.argmax(vals)])
    return k if vals.max() > 0.15 else None


def _periodicity_strength(x: np.ndarray, period: int | None) -> float:
    """Periodicity Strength = ||seasonal||^2 / ||x||^2 after STL decomposition.
    If no period is detected, returns 0.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    x = x.astype(float)
    if period is None or period < 2:
        return 0.0
    try:
        stl = STL(x, period=period, robust=True)
        res = stl.fit()
        num = float(np.sum(res.seasonal**2))
        den = float(np.sum(x**2) + 1e-12)
        return max(0.0, min(1.0, num / den))
    except Exception:
        # Robust to STL failures on short/noisy series
        return 0.0


def load_rdataset(package: str, item: str, value_col: str | None = None) -> TimeSeriesWithMeta:
    """Load a univariate time series from Rdatasets (CSV endpoint).
    Many datasets are rectangular; pick a numeric column via `value_col`.
    If omitted, we take the last numeric column by heuristic.

    Returns a standardized series (zero mean, unit std) plus period & PS.
    """
    url = f"{BASE}/{package}/{item}.csv"
    df = pd.read_csv(url)

    if value_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("No numeric columns found; specify value_col.")
        value_col = numeric_cols[-1]

    x = df[value_col].to_numpy(dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=1)) + 1e-12
    x_std = (x - mu) / sd
    x_std = np.nan_to_num(x_std)

    period = _infer_period_via_acf(x_std)
    ps = _periodicity_strength(x_std, period)
    meta = {"package": package, "item": item, "value_col": value_col, "rows": len(x)}
    return TimeSeriesWithMeta(
        name=f"{item}_ts", values=x_std, frequency=period, periodicity_strength=ps, meta=meta
    )
