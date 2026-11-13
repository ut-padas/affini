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
import time
from pathlib import Path

import pytest

from src.infra import backend
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


def test_multi_rank_agrees_with_serial_in_accuracy(parallel_result, serial_result):
    """
    Distributing the work must not change the model's quality. It does change the exact
    tree: AFINNI's augmentation stream is seeded with ``seed + level + batch + rank*1000``,
    so which points a rejected cluster gains depends on the rank and batch that process it
    (on 2-8 ranks the smoke model has 235-252 leaves). That seeding defines AFINNI's results,
    so this compares accuracy instead: the
    same error bound, an error no worse than serial's to within 25%, and a leaf count within
    15% of serial's. ART, whose streams are keyed by node, is checked for exact
    agreement in test_art_is_independent_of_ranks_and_stealing.
    """
    from tests._smoke import SMOKE_TOL

    assert serial_result["n_ranks"] == 1 and serial_result["has_mpi"] is False
    assert parallel_result["max_rel_err"] <= SMOKE_TOL
    assert parallel_result["max_rel_err"] <= 1.25 * serial_result["max_rel_err"]
    assert abs(parallel_result["n_clusters"] - serial_result["n_clusters"]) <= 0.15 * serial_result["n_clusters"]


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


@pytest.mark.parametrize(
    ("phase", "message"),
    [("root_initial_oracle", "rank 0 (initial evaluation)"),
     ("root_initial_clustering", "rank 0 (initial clustering)"),
     ("worker_refinement", "rank 1 (refinement)"),
     ("worker_collection", "rank 1 (collection)"),
     ("root_public_result", "rank 0 (public result conversion)"),
     ("worker_index", "rank 1 (index construction)")],
)  # fmt: skip
def test_a_failure_on_one_rank_aborts_the_whole_job(mpi_launcher, phase, message):
    """In every phase, the failing rank must end the job: peers would otherwise block until an external timeout."""
    t0 = time.time()
    proc = subprocess.run(
        [*mpi_launcher, "-n", str(max(2, N_RANKS)), sys.executable, "-m", "tests.mpi._abort_worker", phase],
        cwd=REPO_ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert proc.returncode != 0
    assert f"injected failure on {message}" in proc.stdout + proc.stderr
    assert time.time() - t0 < 90  # terminated promptly, not by the subprocess timeout


@pytest.fixture(scope="module")
def art_serial() -> dict:
    return _run_worker([], {"AFINNI_MPI": "off"}, module="tests.mpi._art_worker")


def test_art_is_independent_of_ranks_and_stealing(mpi_launcher, art_serial):
    """Node-keyed random streams: the same tree and predictions on any rank count, with or without steals."""
    parallel = _run_worker([*mpi_launcher, "-n", str(N_RANKS)], module="tests.mpi._art_worker")
    for name, ref in (("no_steal", "no_steal"), ("steal", "no_steal"), ("soft_no_steal", "soft_no_steal"),
                      ("soft_steal", "soft_no_steal"), ("voronoi_no_steal", "voronoi_no_steal"),
                      ("voronoi_steal", "voronoi_no_steal")):  # fmt: skip
        reference = art_serial[ref]
        for key in ("leaves", "keys_hash", "pred_hash", "evals", "statuses", "outside"):
            assert parallel[name][key] == reference[key], (name, key, parallel[name], reference)
    assert parallel["no_steal"]["grants"] == 0
    if N_RANKS > 1:
        assert parallel["steal"]["grants"] > 0, parallel["steal"]
        assert parallel["soft_steal"]["grants"] > 0, parallel["soft_steal"]
