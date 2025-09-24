from __future__ import annotations

from arkan_ts.datasets import load_rdataset
from arkan_ts.grid_search import (
    default_arima_grid,
    default_arkan_grid,
    grid_search_arima,
    grid_search_arkan,
)


def main() -> None:
    for (pkg, item) in [("fma", "a10"), ("datasets", "discoveries")]:
        ts = load_rdataset(pkg, item)  # standardized
        print(f"{ts.name}: period={ts.frequency}, PeriodicityStrength={100*ts.periodicity_strength:.2f}%")

        best_a, mse_a, _ = grid_search_arima(ts.values, default_arima_grid())
        best_k, mse_k, _ = grid_search_arkan(ts.values, default_arkan_grid(basis="bspline"))
        print("  ARIMA best:", best_a, "MSE=", mse_a)
        print("  AR-KAN best:", best_k, "MSE=", mse_k)


if __name__ == "__main__":
    main()
