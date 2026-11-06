"""
AFINNI package root: the active array module ``xp`` (NumPy or CuPy), backend flags,
host/device transfer helpers and project paths, shared by every module.
"""

import logging
from pathlib import Path
from typing import Any

from array_api_compat.common._typing import Array

from src.backend import (
    HAS_CUML,
    IS_CUPY_BACKEND,
    backend_info,
    get_mpi_rank,
    get_mpi_size,
    has_mpi,
    print_backend_info,
    xp,
)

logger = logging.getLogger(__name__)

# Historical names used throughout the codebase.
_get_mpi_rank = get_mpi_rank
_get_mpi_size = get_mpi_size


def __getattr__(name: str) -> Any:
    """Resolve ``HAS_MPI``/``MPI`` on first use, so importing ``src`` never initializes MPI."""
    # Keep `HAS_MPI` / `MPI` lazy (see src.backend) so `import src` never runs MPI_Init.
    if name in ("HAS_MPI", "MPI"):
        from src import backend

        return getattr(backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _to_host(arr: Array | Any) -> Array | Any:
    """Move array to host (NumPy) if it's on device (CuPy)."""

    # If not an array, return as is
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        return arr

    if hasattr(arr, "get"):  # CuPy array
        return arr.get()
    return arr


def _to_device(arr: Array | Any) -> Array | Any:
    """Move array to device (CuPy) if available, else keep as NumPy."""
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        return arr
    return xp.asarray(arr)


# Define project directories
_file_path = Path(__file__).parent.resolve()
PROJECT_DIR = _file_path.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

__all__ = [
    "HAS_CUML",
    "HAS_MPI",
    "IS_CUPY_BACKEND",
    "MPI",
    "Array",
    "FIGURES_DIR",
    "PROJECT_DIR",
    "RESULTS_DIR",
    "_get_mpi_rank",
    "_get_mpi_size",
    "_to_device",
    "_to_host",
    "backend_info",
    "get_mpi_rank",
    "get_mpi_size",
    "has_mpi",
    "print_backend_info",
    "xp",
]
