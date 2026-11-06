"""
Optional-backend detection for AFINNI: GPU (CuPy / cuML) and MPI (mpi4py).

Every optional dependency is probed here, once, and exposed through a small set
of standard flags. The rest of the codebase should only ever consult these
flags (never ``try: import cupy`` / ``from mpi4py import MPI`` directly), so the
same code runs unchanged on a laptop (CPU, no MPI), a workstation GPU, or an
HPC cluster.

Flags
-----
``IS_CUPY_BACKEND``
    True when the array backend ``xp`` is CuPy (a usable CUDA device was found).
``HAS_CUML``
    True when cuML is installed *and* the CuPy backend is active. cuML is
    Linux-only, so on e.g. Windows + GPU this is False: KMeans and kNN fall
    back to scikit-learn / FAISS on the host while everything else stays on
    the GPU.
``HAS_MPI``
    True when mpi4py imported *and* found an MPI library. Resolved lazily on
    first access, so ``import src`` never initializes MPI by itself.
``MPI``
    The real ``mpi4py.MPI`` module when ``HAS_MPI``; otherwise a serial
    stand-in with a single rank (rank 0, size 1) that supports the subset of
    the API used by AFINNI. Code can therefore always write
    ``from src.backend import MPI`` and ``MPI.COMM_WORLD``.

Environment overrides
---------------------
``AFINNI_DEVICE`` = ``auto`` (default) | ``cpu`` | ``gpu``
    ``cpu`` forces NumPy even when a GPU is present; ``gpu`` fails loudly if
    CuPy cannot use a GPU instead of silently falling back.
``AFINNI_MPI`` = ``auto`` (default) | ``off`` | ``on``
    ``off`` uses the serial stand-in even if mpi4py is installed; ``on`` fails
    loudly if MPI is unavailable.

Run ``python -m src`` to print what was detected.
"""

# The serial stand-ins mirror mpi4py's CamelCase API (Get_rank, Iprobe, ...).
# ruff: noqa: N802

from __future__ import annotations

import functools
import importlib.util
import logging
import multiprocessing
import os
import platform
import sys
import time
from typing import Any

import numpy as np
from array_api_compat import array_namespace

logger = logging.getLogger(__name__)

_TRUE = {"1", "on", "true", "yes"}
_FALSE = {"0", "off", "false", "no"}


def _env_choice(var: str, choices: dict[str, set[str]]) -> str:
    """Canonical value of an ``auto``-defaulting environment switch; raises on unknown values."""
    raw = os.environ.get(var, "auto").strip().lower() or "auto"
    for canonical, aliases in choices.items():
        if raw == canonical or raw in aliases:
            return canonical
    allowed = " | ".join(choices)
    raise ValueError(f"{var}={raw!r} is not valid; expected one of: {allowed}")


# =============================================================================
# GPU: CuPy (array backend) and cuML (GPU KMeans / kNN)
# =============================================================================
DEVICE_MODE = _env_choice(
    "AFINNI_DEVICE", {"auto": set(), "cpu": {"numpy", "host"}, "gpu": {"cuda", "cupy"}}
)


def _detect_cupy() -> tuple[Any, str]:
    """Return ``(cupy_module_or_None, human_readable_reason)``."""
    if DEVICE_MODE == "cpu":
        return None, "disabled via AFINNI_DEVICE=cpu"
    try:
        import cupy
    except Exception as e:  # ImportError, or broken CUDA libs raising on import
        return None, f"CuPy not importable ({type(e).__name__}: {e})"
    try:
        if not cupy.is_available():
            return None, "CuPy installed but no usable CUDA device was found"
    except Exception as e:  # driver missing / mismatched
        return None, f"CuPy installed but CUDA is unusable ({type(e).__name__}: {e})"
    return cupy, f"CuPy {cupy.__version__}"


_cupy, GPU_REASON = _detect_cupy()
IS_CUPY_BACKEND: bool = _cupy is not None

if DEVICE_MODE == "gpu" and not IS_CUPY_BACKEND:
    raise RuntimeError(f"AFINNI_DEVICE=gpu was requested but {GPU_REASON}.")

# find_spec avoids paying cuML's multi-second import cost at startup.
HAS_CUML: bool = IS_CUPY_BACKEND and importlib.util.find_spec("cuml") is not None

xp = array_namespace(_cupy.array([0]) if IS_CUPY_BACKEND else np.array([0]))

logger.info(
    "Array backend: %s (%s); cuML: %s",
    "CuPy" if IS_CUPY_BACKEND else "NumPy",
    GPU_REASON if IS_CUPY_BACKEND else f"CPU mode - {GPU_REASON}",
    "yes" if HAS_CUML else "no (scikit-learn / FAISS fallback)",
)


# =============================================================================
# MPI: mpi4py, or a single-rank serial stand-in
# =============================================================================
MPI_MODE = _env_choice("AFINNI_MPI", {"auto": set(), "off": _FALSE, "on": _TRUE})

# Environment variables set by common MPI launchers (Open MPI, MPICH/Hydra,
# Intel MPI, MVAPICH). If one says we are part of a multi-rank job but MPI is
# unusable, silently running serially would make every rank believe it is
# rank 0 and duplicate the whole computation - so we refuse instead.
_LAUNCHER_SIZE_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")


def _launched_with_multiple_ranks() -> bool:
    """True if a launcher started several ranks, read from its environment without initializing MPI."""
    for var in _LAUNCHER_SIZE_VARS:
        try:
            if int(os.environ.get(var, "1")) > 1:
                return True
        except ValueError:
            continue
    return False


def _in_multiprocessing_worker() -> bool:
    """
    True inside a ``multiprocessing`` child (spawn/fork/forkserver pool worker).

    Spawned workers (the default on Windows/macOS) re-import the entry script, so
    module-level ``_get_mpi_size()`` calls there would otherwise run MPI_Init in
    every worker with the parent's launcher environment - which can hang or crash
    under ``mpiexec``. Workers never communicate over MPI, so they get the serial
    stand-in.
    """
    return multiprocessing.current_process().name != "MainProcess"


class _SerialStatus:
    """Stand-in for ``MPI.Status``."""

    def __init__(self) -> None:
        self.source = 0
        self.tag = -1

    def Get_source(self) -> int:
        return self.source

    def Get_tag(self) -> int:
        return self.tag


class _SerialComm:
    """
    Stand-in for ``MPI.COMM_WORLD`` in a single-process run: rank 0 of size 1.

    Collectives are the identity (the root is the only member). Point-to-point
    messages are impossible because there is no peer; AFINNI's algorithms only
    address *other* ranks, so with one rank they are never reached, and calling
    one here indicates a bug, hence the explicit error.
    """

    rank = 0
    size = 1

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def Barrier(self) -> None:
        return None

    barrier = Barrier

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj

    def scatter(self, sendobj: Any, root: int = 0) -> Any:
        return sendobj[0]

    def gather(self, sendobj: Any, root: int = 0) -> list:
        return [sendobj]

    def allgather(self, sendobj: Any) -> list:
        return [sendobj]

    def reduce(self, sendobj: Any, op: Any = None, root: int = 0) -> Any:
        return sendobj

    def allreduce(self, sendobj: Any, op: Any = None) -> Any:
        return sendobj

    def alltoall(self, sendobj: Any) -> list:
        return list(sendobj)

    def Iprobe(self, source: int = -1, tag: int = -1, status: Any = None) -> bool:
        return False  # nobody else can ever send us anything

    def Abort(self, errorcode: int = 1) -> None:
        sys.exit(errorcode)

    def _no_peer(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Point-to-point MPI communication was attempted in a serial (no-MPI) "
            "run, but there is no other rank to talk to. Install an MPI extra "
            "(see README) and launch with mpiexec, or fix the caller to skip "
            "communication when size == 1."
        )

    send = recv = Send = Recv = isend = irecv = Isend = Irecv = _no_peer


class _SerialMPI:
    """Stand-in for the ``mpi4py.MPI`` module (single rank, no communication)."""

    IS_SERIAL_FALLBACK = True

    ANY_SOURCE = -1
    ANY_TAG = -1
    PROC_NULL = -2
    SUM, PROD, MAX, MIN = "SUM", "PROD", "MAX", "MIN"

    Status = _SerialStatus
    Comm = Intracomm = _SerialComm
    COMM_WORLD = _SerialComm()
    COMM_SELF = COMM_WORLD

    @staticmethod
    def Finalize() -> None:
        return None

    @staticmethod
    def Is_initialized() -> bool:
        return True

    @staticmethod
    def Is_finalized() -> bool:
        return False

    @staticmethod
    def Wtime() -> float:
        return time.perf_counter()

    @staticmethod
    def Get_processor_name() -> str:
        return platform.node()


@functools.cache
def _resolve_mpi() -> tuple[Any, bool, str]:
    """Return ``(MPI_module, has_mpi, reason)``. Imports mpi4py at most once."""
    if MPI_MODE == "off":
        return _SerialMPI, False, "disabled via AFINNI_MPI=off"
    if _in_multiprocessing_worker():
        return _SerialMPI, False, "multiprocessing worker (MPI never used in workers)"

    try:
        from mpi4py import MPI as _MPI
    except ImportError:
        reason = "mpi4py not installed"
    except RuntimeError as e:
        # mpi4py wheels load libmpi at import time and raise RuntimeError when
        # no MPI library is found (e.g. `mpi-system` extra but no system MPI).
        first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
        reason = f"mpi4py installed but no MPI library could be loaded ({first_line})"
    else:
        lib = _MPI.Get_library_version().strip().splitlines()[0]
        return _MPI, True, f"mpi4py with {lib}"

    if MPI_MODE == "on":
        raise RuntimeError(f"AFINNI_MPI=on was requested but {reason}.")
    if _launched_with_multiple_ranks():
        raise RuntimeError(
            f"This process was started by an MPI launcher with more than one rank, "
            f"but MPI is unavailable: {reason}. Every rank would run the full job "
            f"as rank 0. Install an MPI extra (see README) or run without mpiexec."
        )
    logger.info("MPI unavailable (%s); running serially as rank 0 of 1.", reason)
    return _SerialMPI, False, reason


def get_mpi() -> Any:
    """The ``mpi4py.MPI`` module, or the serial stand-in if MPI is unavailable."""
    return _resolve_mpi()[0]


def has_mpi() -> bool:
    """True if real MPI is in use (resolved, and initialized, on first call)."""
    return _resolve_mpi()[1]


def get_mpi_rank() -> int:
    """Rank of this process in ``COMM_WORLD`` (0 when running without MPI)."""
    return get_mpi().COMM_WORLD.Get_rank()


def get_mpi_size() -> int:
    """Number of ranks in ``COMM_WORLD`` (1 when running without MPI)."""
    return get_mpi().COMM_WORLD.Get_size()


def __getattr__(name: str) -> Any:
    # PEP 562: resolve MPI lazily so importing this module never calls MPI_Init.
    if name == "MPI":
        return get_mpi()
    if name == "HAS_MPI":
        return has_mpi()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# =============================================================================
# Diagnostics
# =============================================================================
def backend_info() -> dict[str, Any]:
    """Machine-readable summary of the detected backends (handy for result JSONs)."""
    _, mpi_ok, mpi_reason = _resolve_mpi()
    info: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device_mode": DEVICE_MODE,
        "is_cupy_backend": IS_CUPY_BACKEND,
        "gpu": GPU_REASON,
        "has_cuml": HAS_CUML,
        "mpi_mode": MPI_MODE,
        "has_mpi": mpi_ok,
        "mpi": mpi_reason,
        "mpi_rank": get_mpi_rank(),
        "mpi_size": get_mpi_size(),
    }
    if IS_CUPY_BACKEND:
        try:
            props = _cupy.cuda.runtime.getDeviceProperties(_cupy.cuda.Device().id)
            info["gpu_name"] = props["name"].decode()
        except Exception:
            pass
    return info


def print_backend_info() -> None:
    """Print ``backend_info()`` as an aligned table (``python -m src``)."""
    info = backend_info()
    width = max(map(len, info))
    print("AFINNI backend configuration")
    for key, value in info.items():
        print(f"  {key:<{width}} : {value}")
