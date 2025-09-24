"""Top-level package for AR-KAN Time Series."""

from .basis import UnivariateBSpline, UnivariateRBFSpline
from .datasets import TimeSeriesWithMeta, load_rdataset
from .grid_search import (
    ARIMAConfig,
    ARKANConfig,
    default_arima_grid,
    default_arkan_grid,
    grid_search_arima,
    grid_search_arkan,
)
from .kan import KANBlock
from .memory import ARMemory, ARMemoryConfig, autocovariance, yule_walker_ar
from .model import ARKAN
from .train import SupervisedTS, train_arkan

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
