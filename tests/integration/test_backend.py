"""
Backend detection (src/backend.py) and an end-to-end smoke test.

These run on every machine. Assertions are relative to what is installed, so
the same suite passes on a CPU-only laptop and on a GPU/MPI node. To turn
"whatever is available" into a hard requirement, set the overrides:

    AFINNI_DEVICE=gpu uv run pytest tests/integration/test_backend.py   # must run on the GPU
    AFINNI_MPI=on     uv run pytest tests/integration/test_backend.py   # mpi4py must load an MPI library

The pytest header shows which backends were actually exercised.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import backend
from tests._smoke import SMOKE_TOL, run_smoke

REPO_ROOT = Path(__file__).resolve().parents[2]

# Variables that influence detection; stripped so subprocess cases are hermetic.
_DETECTION_VARS = ("AFINNI_DEVICE", "AFINNI_MPI", *backend._LAUNCHER_SIZE_VARS)

_PROBE = (
    "import json, src, src.backend as b; "
    "print(json.dumps({'cupy': b.IS_CUPY_BACKEND, 'cuml': b.HAS_CUML, "
    "'xp': b.xp.__name__, 'mpi': b.has_mpi(), 'size': b.get_mpi_size(), "
    "'reason': b._resolve_mpi()[2]}))"
)
# Simulates "mpi4py not installed" regardless of what is in the venv.
_BLOCK_MPI4PY = "import sys; sys.modules['mpi4py'] = None; "


def _run(env: dict[str, str] | None = None, block_mpi4py: bool = False):
    child_env = {k: v for k, v in os.environ.items() if k not in _DETECTION_VARS}
    child_env.update(env or {})
    code = (_BLOCK_MPI4PY if block_mpi4py else "") + _PROBE
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else None
    return proc, result


# --- Flags in this process ---------------------------------------------------
def test_flags_are_consistent():
    assert (backend.xp.__name__.split(".")[-1] == "cupy") == backend.IS_CUPY_BACKEND
    if backend.HAS_CUML:
        assert backend.IS_CUPY_BACKEND, "HAS_CUML must imply the CuPy backend"
    assert backend.get_mpi_size() >= 1
    assert 0 <= backend.get_mpi_rank() < backend.get_mpi_size()
    assert backend.has_mpi() == (not getattr(backend.get_mpi(), "IS_SERIAL_FALLBACK", False))


def test_requested_backends_are_honoured():
    """Hard checks when the tester explicitly asked for a backend."""
    if backend.DEVICE_MODE == "gpu":
        assert backend.IS_CUPY_BACKEND
    if backend.DEVICE_MODE == "cpu":
        assert not backend.IS_CUPY_BACKEND
    if backend.MPI_MODE == "on":
        assert backend.has_mpi(), backend._resolve_mpi()[2]
    if backend.MPI_MODE == "off":
        assert not backend.has_mpi()


def test_backend_info_reports_everything():
    info = backend.backend_info()
    for key in ("platform", "is_cupy_backend", "has_cuml", "has_mpi", "mpi", "mpi_size"):
        assert key in info


# --- Environment overrides (fresh interpreter per case) -----------------------
def test_force_cpu():
    proc, r = _run({"AFINNI_DEVICE": "cpu"})
    assert proc.returncode == 0, proc.stderr
    assert r["cupy"] is False and r["cuml"] is False
    assert r["xp"].split(".")[-1] == "numpy"


def test_force_gpu_matches_hardware():
    if backend.DEVICE_MODE == "cpu":
        pytest.skip("AFINNI_DEVICE=cpu hides whether a GPU is present")
    proc, _ = _run({"AFINNI_DEVICE": "gpu"})
    if backend.IS_CUPY_BACKEND:
        assert proc.returncode == 0, proc.stderr
    else:
        assert proc.returncode != 0
        assert "AFINNI_DEVICE=gpu was requested" in proc.stderr


@pytest.mark.parametrize("var", ["AFINNI_DEVICE", "AFINNI_MPI"])
def test_invalid_override_is_rejected(var):
    proc, _ = _run({var: "bogus"})
    assert proc.returncode != 0
    assert "is not valid" in proc.stderr


def test_mpi_off_uses_serial_stand_in():
    proc, r = _run({"AFINNI_MPI": "off"})
    assert proc.returncode == 0, proc.stderr
    assert r["mpi"] is False and r["size"] == 1
    assert "AFINNI_MPI=off" in r["reason"]


def test_without_mpi4py_falls_back_to_serial():
    proc, r = _run(block_mpi4py=True)
    assert proc.returncode == 0, proc.stderr
    assert r["mpi"] is False and r["size"] == 1


def test_without_mpi4py_mpi_on_fails_loudly():
    proc, _ = _run({"AFINNI_MPI": "on"}, block_mpi4py=True)
    assert proc.returncode != 0
    assert "AFINNI_MPI=on was requested" in proc.stderr


@pytest.mark.parametrize("launcher_var", backend._LAUNCHER_SIZE_VARS)
def test_without_mpi_under_multi_rank_launcher_fails_loudly(launcher_var):
    """Every rank silently running the whole job as rank 0 must be impossible."""
    proc, _ = _run({launcher_var: "4"}, block_mpi4py=True)
    assert proc.returncode != 0
    assert "started by an MPI launcher" in proc.stderr


# --- Serial stand-in ------------------------------------------------------------
def test_serial_comm_semantics():
    comm = backend._SerialMPI.COMM_WORLD
    assert (comm.Get_rank(), comm.Get_size()) == (0, 1)
    assert comm.bcast({"a": 1}, root=0) == {"a": 1}
    assert comm.scatter([np.arange(3)], root=0).tolist() == [0, 1, 2]
    assert comm.gather(7, root=0) == [7]
    assert comm.allreduce(5, op=backend._SerialMPI.SUM) == 5
    assert comm.Iprobe(source=backend._SerialMPI.ANY_SOURCE, tag=1) is False
    assert comm.Iprobe() is False
    assert backend._SerialMPI.Status().Get_source() == 0
    comm.Barrier()
    backend._SerialMPI.Finalize()


@pytest.mark.parametrize("method", ["send", "recv", "Send", "Recv"])
def test_serial_comm_rejects_point_to_point(method):
    with pytest.raises(RuntimeError, match="no other rank"):
        getattr(backend._SerialMPI.COMM_WORLD, method)(np.zeros(1), 0)


# --- Multiprocessing (spawn = default on Windows/macOS) ----------------------------
def _worker_mpi_state(_):
    from src import backend as b

    _, has_mpi, reason = b._resolve_mpi()
    return has_mpi, reason


def test_spawn_workers_never_initialise_mpi(monkeypatch):
    # The worker must decide on its own (AFINNI_MPI=off would short-circuit first).
    monkeypatch.setenv("AFINNI_MPI", "auto")
    with mp.get_context("spawn").Pool(1) as pool:
        has_mpi, reason = pool.map(_worker_mpi_state, [0])[0]
    assert has_mpi is False
    assert "multiprocessing worker" in reason


# --- End-to-end -----------------------------------------------------------------------
def test_surrogate_smoke():
    """Train + predict in whatever backend is active (CPU, CPU-without-cuML GPU, or cuML)."""
    result = run_smoke()
    assert result is not None
    assert result["n_clusters"] > 0
    # Test points can fall slightly outside the trained trust regions, so allow
    # headroom over the training tolerance; a broken backend is off by orders of magnitude.
    assert result["max_rel_err"] < 1.5 * SMOKE_TOL, result
    assert result["is_cupy_backend"] == backend.IS_CUPY_BACKEND
