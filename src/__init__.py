"""
AFINNI package root: the active array module ``xp`` (NumPy or CuPy), backend flags,
host/device transfer helpers and project paths, shared by every module.
"""

import logging
from pathlib import Path
from typing import Any

from array_api_compat.common._typing import Array

from src.infra.backend import (
    IS_CUPY_BACKEND,
    backend_info,
    get_mpi_rank,
    get_mpi_size,
    has_mpi,
    print_backend_info,
    xp,
)

logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    """Resolve ``HAS_MPI``/``MPI`` on first use, so importing ``src`` never initializes MPI."""
    # Keep `HAS_MPI` / `MPI` lazy (see src.infra.backend) so `import src` never runs MPI_Init.
    if name in ("HAS_MPI", "MPI"):
        from src.infra import backend

        return getattr(backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def to_host(arr: Array | Any) -> Array | Any:
    """A NumPy copy of a CuPy array; NumPy arrays and non-arrays are returned as they are."""
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        return arr

    if hasattr(arr, "get"):  # CuPy array
        return arr.get()
    return arr


def to_device(arr: Array | Any) -> Array | Any:
    """The array on the active backend (CuPy when ``xp`` is CuPy); non-arrays are returned as they are."""
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        return arr
    return xp.asarray(arr)


# Define project directories
_file_path = Path(__file__).parent.resolve()
PROJECT_DIR = _file_path.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

__all__ = [
    "Array",
    "backend_info",
    "FIGURES_DIR",
    "get_mpi_rank",
    "get_mpi_size",
    "HAS_MPI",
    "has_mpi",
    "IS_CUPY_BACKEND",
    "MPI",
    "print_backend_info",
    "PROJECT_DIR",
    "RESULTS_DIR",
    "to_device",
    "to_host",
    "xp",
]
