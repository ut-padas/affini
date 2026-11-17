"""
Package root: the active array module ``xp`` (NumPy or CuPy), backend flags,
host/device transfer helpers and project paths, shared by every module.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
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


# Bytes of the reused page-locked buffer that copy_to_host stages downloads through. A copy into
# pageable memory runs at a fraction of the bus rate (the driver stages it itself and the
# destination pages fault in); larger arrays go through in pieces of this size.
PINNED_STAGING_BYTES = 64 << 20
_pinned_staging: np.ndarray | None = None


def copy_to_host(arr: Array | Any, out: np.ndarray) -> None:
    """
    Write ``arr`` into the host array ``out``, bit for bit as ``out[...] = to_host(arr)``. A CuPy
    array with ``out``'s shape and dtype and a C-contiguous ``out`` is copied through one reused
    pinned buffer of ``PINNED_STAGING_BYTES``, in pieces when larger; anything else takes
    ``to_host``. Each piece's download blocks, so the buffer is free again when it returns.
    """
    global _pinned_staging
    if (not hasattr(arr, "get") or arr.shape != out.shape or arr.dtype != out.dtype
            or not out.flags.c_contiguous or arr.size == 0):  # fmt: skip
        out[...] = to_host(arr)
        return
    if _pinned_staging is None:
        import cupy as cp

        _pinned_staging = np.frombuffer(cp.cuda.alloc_pinned_memory(PINNED_STAGING_BYTES), np.uint8)
    src, dst = xp.ascontiguousarray(arr).reshape(-1), out.reshape(-1)
    stage = _pinned_staging[: PINNED_STAGING_BYTES // arr.itemsize * arr.itemsize].view(arr.dtype)
    for s in range(0, int(src.size), int(stage.size)):
        e = min(s + int(stage.size), int(src.size))
        src[s:e].get(out=stage[: e - s])
        dst[s:e] = stage[: e - s]


def release_pinned_staging() -> None:
    """Return ``copy_to_host``'s pinned buffer, and the pinned blocks CuPy cached, to the system."""
    global _pinned_staging
    if _pinned_staging is not None:
        _pinned_staging = None
        import cupy as cp

        cp.get_default_pinned_memory_pool().free_all_blocks()


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
    "copy_to_host",
    "FIGURES_DIR",
    "get_mpi_rank",
    "get_mpi_size",
    "HAS_MPI",
    "has_mpi",
    "IS_CUPY_BACKEND",
    "MPI",
    "print_backend_info",
    "PROJECT_DIR",
    "release_pinned_staging",
    "RESULTS_DIR",
    "to_device",
    "to_host",
    "xp",
]
