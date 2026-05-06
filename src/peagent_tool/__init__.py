"""Top-level package for PEAgent."""

from importlib.metadata import PackageNotFoundError, version

from .celltype_prediction import CellTypePredictor, list_valid_celltypes, load_predictor, predict_celltypes

__author__ = """Jie Yao"""
__email__ = "JieYao@uga.edu"
try:
    __version__ = version("peagent-tool")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CellTypePredictor",
    "__version__",
    "list_valid_celltypes",
    "load_predictor",
    "predict_celltypes",
]
