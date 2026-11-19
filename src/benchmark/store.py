"""
Where the benchmark keeps what it measures, for every suite and every method (AFINNI, ART and the schemes).

    results/benchmark/runs/<suite>/<setup>/<problem>/d<d>/<method>/tol<t>-s<seed>.json   one run's record
                                                                         ...-s<seed>.npz   its arrays (optional)
    results/benchmark/data/<suite>/<problem>/<name>      a problem's shared inputs (trajectories, baselines)
    figures/benchmark/<suite>/<setup>/                   the plots, redrawn from the runs alone

``results/`` and ``figures/`` are ``src.RESULTS_DIR`` and ``src.FIGURES_DIR`` (not committed). A setup names
the device and the rank count (``cpu-8rank``): times depend on both, and ART's split decisions differ between
CPU and GPU, so one comparison never mixes setups.

A record is a JSON object. Every suite writes the same identifying fields (``suite``, ``problem``, ``d``,
``method``, ``tol``, ``seed``, ``setup``) and, when the run finished, ``evals``, ``train_s``, ``predict_us``
and ``worst_ratio`` (the worst error over the tolerance); a run that did not finish has ``failed`` (the
reason) instead. Each suite adds its own fields. ``recorded`` is the time of writing and ``commit`` the code
it ran (``git describe``; None outside a checkout), so a record can be told apart from the current code's.
Writes are atomic (a temporary file renamed into place), so a reader never sees half a record.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from src import FIGURES_DIR, PROJECT_DIR, RESULTS_DIR

ROOT = RESULTS_DIR / "benchmark"
FIGURES = FIGURES_DIR / "benchmark"
IDENTITY = ("suite", "problem", "d", "method", "tol", "seed", "setup")


@cache
def commit() -> str | None:
    """The checked-out code, ``git describe --always --dirty`` (None outside a git checkout)."""
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=PROJECT_DIR, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def atomic_write(path: Path, write) -> None:
    """``write(tmp_path)`` then rename into ``path``: concurrent readers and writers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}-{os.getpid()}.tmp{path.suffix}")
    write(tmp)
    tmp.replace(path)


def write_json(path: Path, obj: dict) -> None:
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(obj)))


@dataclass(frozen=True)
class Store:
    """The runs, data and figures of one suite on one setup."""

    suite: str
    setup: str
    root: Path = ROOT
    figures_root: Path = FIGURES

    def run_path(self, problem: str, d: int, method: str, tol: float, seed: int) -> Path:
        return self.root / "runs" / self.suite / self.setup / problem / f"d{d}" / method / f"tol{tol:g}-s{seed}.json"

    def exists(self, problem: str, d: int, method: str, tol: float, seed: int) -> bool:
        return self.run_path(problem, d, method, tol, seed).exists()

    def read(self, problem: str, d: int, method: str, tol: float, seed: int) -> dict | None:
        """The run's record, or None if it was never recorded."""
        path = self.run_path(problem, d, method, tol, seed)
        return json.loads(path.read_text()) if path.exists() else None

    def arrays(self, problem: str, d: int, method: str, tol: float, seed: int) -> dict:
        """The run's arrays (empty if it has none)."""
        path = self.run_path(problem, d, method, tol, seed).with_suffix(".npz")
        if not path.exists():
            return {}
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def write(self, record: dict, arrays: dict | None = None) -> Path:
        """
        Record one run (``record`` holds its identity: problem, d, method, tol, seed); the arrays first, so
        a record on disk always has its arrays.
        """
        record = {"suite": self.suite, "setup": self.setup, **record, "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit()}
        path = self.run_path(record["problem"], int(record["d"]), record["method"], float(record["tol"]), int(record["seed"]))
        if arrays:
            atomic_write(path.with_suffix(".npz"), lambda tmp: np.savez_compressed(tmp, **arrays))
        write_json(path, record)
        return path

    def runs(self, problem: str | None = None, method: str | None = None) -> list[dict]:
        """Every record of this suite and setup (optionally of one problem or method)."""
        base = self.root / "runs" / self.suite / self.setup
        pattern = f"{problem or '*'}/d*/{method or '*'}/tol*-s*.json"
        return [json.loads(p.read_text()) for p in sorted(base.glob(pattern))]

    def data(self, problem: str, name: str) -> Path:
        """A file of the problem's shared inputs (shared by every setup: they depend on the problem alone)."""
        return self.root / "data" / self.suite / problem / name

    @property
    def figures(self) -> Path:
        path = self.figures_root / self.suite / self.setup
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log(self) -> Path:
        path = self.root / "logs" / f"{self.suite}-{self.setup}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
