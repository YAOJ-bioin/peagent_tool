"""Top-level package for PEAgent."""

from importlib.metadata import PackageNotFoundError, version

from .celltype_prediction import CellTypePredictor, list_valid_celltypes, load_predictor, predict_celltypes
from .ism import (
    ISMAttributionResult,
    ISMAttributor,
    compute_and_plot_ism_attribution,
    compute_ism_attribution,
    plot_ism_attribution,
)

__author__ = """Jie Yao"""
__email__ = "JieYao@uga.edu"
try:
    __version__ = version("peagent-tool")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CellTypePredictor",
    "ISMAttributionResult",
    "ISMAttributor",
    "__version__",
    "compute_and_plot_ism_attribution",
    "compute_ism_attribution",
    "list_valid_celltypes",
    "load_predictor",
    "plot_ism_attribution",
    "predict_celltypes",
]
