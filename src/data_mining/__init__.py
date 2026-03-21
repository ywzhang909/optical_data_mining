"""Top-level package for optical data mining.

This package groups fitting, image processing, and analysis utilities for
laser/optical beam data workflows.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("data-miner")
except PackageNotFoundError:  # pragma: no cover - local editable install
    __version__ = "0.0.0"

__all__ = ["__version__"]
