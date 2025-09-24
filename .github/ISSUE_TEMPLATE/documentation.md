# Documentation

This repository implements AR-KAN for time series: an AR(p) memory module that feeds a KAN-style nonlinear stage. It includes grid-search utilities (ARIMA & AR-KAN), STL-based periodicity strength, and an Rdatasets loader.

## 1\. Installation

**Poetry (recommended)**

```bash
poetry install
```

**pip (editable)**

```bash
pip install -e .
```

## 2\. Architecture Overview

- **ARMemory**: estimates AR(p) coefficients via Yule–Walker/Levinson–Durbin and emits AR-weighted lags.

  - Input: 1D series `x` (standardized by default).
  - Output: matrix `Z` of shape `(T-p, p)` with `Z[t,i] = a[i] * x_{t + p - 1 - i}`.

- **KANBlock**: applies a learnable univariate basis per input (RBF or cubic B-spline), then a linear mix.
- **ARKAN**: two KAN blocks with a nonlinearity in-between (`SiLU`), mapping `(B,p) -> (B,1)`.
- **Training**: MSE loss + Adam. 80/20 split used in examples.
- **Datasets**: Rdatasets CSV loader, STL-based periodicity strength, ACF-based period inference.
- **Grid Search**: compact, cache-aware search spaces for ARIMA `(p,d,q)` and AR-KAN `(p,hidden,n_basis,basis)`.

## 3\. Quick Start (Python)

```python
import math, numpy as np
from torch.utils.data import DataLoader
from arkan_ts.memory import ARMemory, ARMemoryConfig
from arkan_ts.model import ARKAN
from arkan_ts.train import SupervisedTS, train_arkan

rng = np.random.default_rng(0)
N = 500
t = np.linspace(0, 8*math.pi, N)
x = np.cos(2*t) + np.cos(2*math.pi*t) + rng.normal(0, 0.2, N)

Ntr = int(0.8*N)
mem = ARMemory(ARMemoryConfig(p=20, standardize=True)).fit(x[:Ntr])
Ztr, ytr = mem.transform(x[:Ntr]), mem.target(x[:Ntr])
Zte, yte = mem.transform(x)[Ntr-20:], x[20:][Ntr-20:]

model = ARKAN(p=20, hidden=50, n_basis=8, basis="bspline")
tr = DataLoader(SupervisedTS(Ztr, ytr), batch_size=64, shuffle=True)
te = DataLoader(SupervisedTS(Zte, yte), batch_size=64)
_, hist = train_arkan(model, tr, te, epochs=200, lr=3e-3, device="cpu")
print("Test MSE:", hist[-1])
```

## 4\. CLI

```bash
# Synthetic demo
poetry run arkan demo --basis bspline --epochs 200

# Grid-search on synthetic f1
poetry run arkan grid --basis bspline --epochs 200
```

## 5\. API Reference (short)

- `arkan_ts.memory`

  - `ARMemoryConfig(p: int, standardize: bool)`
  - `ARMemory.fit(x) -> self`, `.transform(x) -> Z`, `.target(x) -> y`
  - `yule_walker_ar(x, p) -> (a, noise_var)`, `autocovariance(x, lag)`

- `arkan_ts.basis`

  - `UnivariateRBFSpline(n_basis, init_span)`
  - `UnivariateBSpline(n_basis, init_span, degree=3)`

- `arkan_ts.kan`

  - `KANBlock(in_dim, out_dim, n_basis, basis={rbf,bspline})`

- `arkan_ts.model`

  - `ARKAN(p, hidden=50, n_basis=8, basis="rbf"|"bspline")`

- `arkan_ts.train`

  - `SupervisedTS(X, y)`
  - `train_arkan(model, train_loader, test_loader, epochs, lr, device) -> (train_mse[], test_mse[])`

- `arkan_ts.grid_search`

  - `ARIMAConfig, ARKANConfig`
  - `grid_search_arima(series, grid) -> (best_cfg, best_mse, report)`
  - `grid_search_arkan(series, grid, ...) -> (best_cfg, best_mse, report)`
  - `default_arima_grid()`, `default_arkan_grid(basis)`

- `arkan_ts.datasets`

  - `load_rdataset(package, item, value_col=None) -> TimeSeriesWithMeta`

## 6\. Periodicity Strength (STL)

- **Period** selection: argmax of ACF for lags > 0, thresholded (default > 0.15).
- **Strength**: `||seasonal||^2 / ||x||^2` from `statsmodels.tsa.seasonal.STL` decomposition.

## 7\. Hyperparameter Cheatsheet

- `p`: {10, 20, 30}
- `hidden`: {25, 50, 100}
- `n_basis`: {4, 8, 12}
- `basis`: `rbf` or `bspline`
- ARIMA grid: `(p,d,q) ∈ {10,20,30} × {0,1} × {0,1,2}`

## 8\. Reproducibility

- Fix `numpy` RNG via `np.random.default_rng(seed)`.
- For PyTorch, set `torch.manual_seed(seed)` and `torch.use_deterministic_algorithms(True)` (if needed).
- Keep `standardize=True` in `ARMemory` for stability.

## 9\. Versioning

- Semantic-ish: minor bumps for features, patch for bugfixes.
- See `CITATION.cff` for the current version and citation.
