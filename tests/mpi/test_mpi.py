"""
Multi-rank tests: launch tests/mpi/_mpi_worker.py under a real MPI launcher.

Skipped when no MPI is available (CPU-only install). Make them mandatory with

    AFINNI_MPI=on uv run pytest tests/mpi

Configuration (environment):
    AFINNI_MPIEXEC   launcher command, default "mpiexec" (e.g. "ibrun", "srun")
    AFINNI_MPI_NP    number of ranks, default 2
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src import backend
from tests.mpi._mpi_worker import RESULT_PREFIX

pytestmark = pytest.mark.mpi

REPO_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 600
N_RANKS = int(os.environ.get("AFINNI_MPI_NP", "2"))


def _launcher() -> list[str]:
    cmd = shlex.split(os.environ.get("AFINNI_MPIEXEC", "mpiexec"))
    # Prefer the launcher shipped inside the venv (the `mpi` extra) when present.
    # MPICH (Linux/macOS) installs it next to python; Intel MPI's impi-rt
    # (Windows) uses the conda layout, i.e. .venv\Library\bin.
    for bin_dir in (Path(sys.executable).parent, Path(sys.prefix) / "Library" / "bin"):
        venv_bin = bin_dir / cmd[0]
        for candidate in (venv_bin, venv_bin.with_suffix(".exe")):
            if candidate.exists():
                return [str(candidate), *cmd[1:]]
    found = shutil.which(cmd[0])
    return [found, *cmd[1:]] if found else []


@pytest.fixture(scope="module")
def mpi_launcher() -> list[str]:
    strict = backend.MPI_MODE == "on"
    problem = None
    if not backend.has_mpi():
        problem = f"MPI unavailable: {backend._resolve_mpi()[2]}"
    elif not _launcher():
        problem = f"MPI launcher {os.environ.get('AFINNI_MPIEXEC', 'mpiexec')!r} not found on PATH"
    if problem:
        if strict:
            pytest.fail(f"AFINNI_MPI=on but {problem}")
        pytest.skip(problem)
    return _launcher()


def _run_worker(
    cmd: list[str],
    env_overrides: dict[str, str] | None = None,
    module: str = "tests.mpi._mpi_worker",
) -> dict:
    env = dict(os.environ)
    env.update(env_overrides or {})
    try:
        proc = subprocess.run(
            [*cmd, sys.executable, "-m", module],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"MPI worker did not finish within {TIMEOUT_S}s (deadlock?): {cmd}")
    assert proc.returncode == 0, f"worker failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(RESULT_PREFIX)]
    assert len(lines) == 1, f"expected one result line from rank 0, got:\n{proc.stdout}"
    return json.loads(lines[0][len(RESULT_PREFIX):])


@pytest.fixture(scope="module")
def serial_result() -> dict:
    """Reference run in a single process with MPI disabled."""
    return _run_worker([], {"AFINNI_MPI": "off"})


@pytest.fixture(scope="module")
def parallel_result(mpi_launcher) -> dict:
    return _run_worker([*mpi_launcher, "-n", str(N_RANKS)])


def test_multi_rank_run_succeeds(parallel_result):
    assert parallel_result["has_mpi"] is True
    assert parallel_result["n_ranks"] == N_RANKS
    assert parallel_result["n_clusters"] > 0


def test_multi_rank_matches_serial(parallel_result, serial_result):
    """Distributing the work across ranks must not change the trained model."""
    assert serial_result["n_ranks"] == 1 and serial_result["has_mpi"] is False
    assert parallel_result["n_clusters"] == serial_result["n_clusters"]
    assert parallel_result["max_rel_err"] == pytest.approx(serial_result["max_rel_err"], rel=1e-6)


def test_work_stealing_conserves_work_and_drains_messages(mpi_launcher):
    result = _run_worker(
        [*mpi_launcher, "-n", str(N_RANKS)],
        module="tests.mpi._work_stealing_worker",
    )
    assert result["conserved"] is True


def test_work_stealing_end_to_end_leaves_no_messages(mpi_launcher):
    """Real refinement must grant work at a level boundary and finish cleanly."""
    result = _run_worker(
        [*mpi_launcher, "-n", str(N_RANKS)],
        module="tests.mpi._steal_e2e_worker",
    )
    assert result["ok"] is True, result["runs"]
    assert result["grants"] > 0, result["runs"]


def test_work_stealing_moves_gradients_and_keeps_second_order_models_exact(mpi_launcher):
    """Steals of gradient-carrying work are complete and do not change the exact result."""
    result = _run_worker(
        [*mpi_launcher, "-n", str(N_RANKS)],
        module="tests.mpi._grad_steal_worker",
    )
    assert result["ok"] is True, result
