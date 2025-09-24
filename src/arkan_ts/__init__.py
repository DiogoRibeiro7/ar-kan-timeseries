"""Top-level package for AR-KAN Time Series."""

from .memory import ARMemory, ARMemoryConfig, yule_walker_ar, autocovariance
from .basis import UnivariateRBFSpline, UnivariateBSpline
from .kan import KANBlock
from .model import ARKAN
from .train import SupervisedTS, train_arkan
from .grid_search import (
    ARIMAConfig,
    ARKANConfig,
    grid_search_arima,
    grid_search_arkan,
    default_arima_grid,
    default_arkan_grid,
)
from .datasets import TimeSeriesWithMeta, load_rdataset

__all__ = [
    "ARMemory",
    "ARMemoryConfig",
    "yule_walker_ar",
    "autocovariance",
    "UnivariateRBFSpline",
    "UnivariateBSpline",
    "KANBlock",
    "ARKAN",
    "SupervisedTS",
    "train_arkan",
    "ARIMAConfig",
    "ARKANConfig",
    "grid_search_arima",
    "grid_search_arkan",
    "default_arima_grid",
    "default_arkan_grid",
    "TimeSeriesWithMeta",
    "load_rdataset",
]

__version__ = "0.1.0"
__author__ = "Diogo Ribeiro"