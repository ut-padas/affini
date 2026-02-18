import logging
import warnings
from pathlib import Path

import numpy as np
from array_api_compat import array_namespace
from array_api_compat.common._typing import Array

logger = logging.getLogger(__name__)


def get_mpi_rank() -> int:
    """Return current MPI rank, or 0 if MPI is not in use."""
    try:
        from mpi4py import MPI as _MPI

        return _MPI.COMM_WORLD.Get_rank()
    except ImportError:
        return 0


def get_mpi_size() -> int:
    """Return total number of MPI ranks, or 1 if MPI is not in use."""
    try:
        from mpi4py import MPI as _MPI

        return _MPI.COMM_WORLD.Get_size()
    except ImportError:
        return 1


try:
    import cupy as cp

    IS_CUPY_BACKEND = cp.is_available()
except Exception:
    IS_CUPY_BACKEND = False
    if get_mpi_rank() == 0:
        msg = "CuPy not available; GPU acceleration will be disabled."
        warnings.warn(msg)

# Set up the array namespace based on backend availability
if IS_CUPY_BACKEND:
    _dummy = cp.array([0])
    logging.info("Array backend: CuPy")
else:
    _dummy = np.array([0])
    logging.info("Array backend: NumPy")

xp = array_namespace(_dummy)


def _to_host(arr: Array) -> Array:
    """Move array to host (NumPy) if it's on device (CuPy)."""
    if hasattr(arr, "get"):  # CuPy array
        return xp.asnumpy(arr)
    return arr


def _to_device(arr: Array) -> Array:
    """Move array to device (CuPy) if available, else keep as NumPy."""
    return xp.asarray(arr)


_file_path = Path(__file__).parent.resolve()
PROJECT_DIR = _file_path.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"
