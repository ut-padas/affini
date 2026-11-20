"""
Where the benchmark keeps what it measures, for every suite and every method (AFINNI, ART and the schemes).

    results/benchmark/runs/<suite>/<setup>/<problem>/d<d>/<method>/tol<t>-s<seed>.json   one run's record
                                                                         ...-s<seed>.npz   its arrays (optional)
    results/benchmark/data/<suite>/<problem>/<name>      a problem's shared inputs (trajectories, baselines)
    figures/benchmark/<suite>/<setup>/                   the plots, redrawn from the runs alone

``results/`` and ``figures/`` are ``src.RESULTS_DIR`` and ``src.FIGURES_DIR`` (not committed). A setup names
the device and the rank count (``cpu-8rank``): times depend on both, and ART's split decisions differ between
CPU and GPU, so one comparison never mixes setups.

A record is a JSON object written through ``src.infra.records``, which adds its envelope: ``schema``, ``status``,
``recorded`` and ``provenance`` (the commit, the code's hash, the machine and the library versions), and
stores the identity under ``key``. Every suite writes the same identifying fields (``suite``, ``problem``,
``d``, ``method``, ``tol``, ``seed``, ``setup``) and, when the run finished, ``evals``, ``train_s``,
``predict_us`` and ``worst_ratio`` (the worst error over the tolerance), ``oracle_s`` (rank 0's time inside
the oracle while training), ``peak_host_mb`` (every rank's peak above its start) and ``model_bytes`` (the
saved model's size); a run that did not finish has ``failed`` (the reason) and, when it raised, ``failure``
(type, message, traceback) instead. Each suite adds its own fields. ``commit`` is ``git describe`` of the code
that ran. Writes are atomic (a temporary file renamed into place), so a reader never sees half a record.
Records of an older schema are upgraded when read (``records.upgrade``); ``python -m src.benchmark.store
migrate`` rewrites every record of the store in the current schema.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import FIGURES_DIR, RESULTS_DIR
from src.infra.records import (
    RecordStore,
    now,
    provenance,
    read_json,
    upgrade,
    write_json,
)

ROOT = RESULTS_DIR / "benchmark"
FIGURES = FIGURES_DIR / "benchmark"
IDENTITY = ("suite", "problem", "d", "method", "tol", "seed", "setup")


def commit() -> str | None:
    """The checked-out code, ``git describe --always --dirty`` (None outside a git checkout)."""
    return provenance()["describe"]


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

    def records(self) -> RecordStore:
        """The record store of this suite and setup, laid out by ``run_path``."""
        return RecordStore(self.root / "runs" / self.suite / self.setup,
                           lambda k: self.run_path(k["problem"], k["d"], k["method"], k["tol"], k["seed"]))  # fmt: skip

    def read(self, problem: str, d: int, method: str, tol: float, seed: int) -> dict | None:
        """The run's record, or None if it was never recorded."""
        return self.records().read(self.key(problem, d, method, tol, seed))

    def key(self, problem: str, d: int, method: str, tol: float, seed: int) -> dict:
        return {"suite": self.suite, "setup": self.setup, "problem": problem, "d": int(d), "method": method, "tol": float(tol), "seed": int(seed)}

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
        key = self.key(record["problem"], record["d"], record["method"], record["tol"], record["seed"])
        self.records().write(key, {"suite": self.suite, "setup": self.setup, **record, "commit": commit()}, arrays)
        return self.records().path(key)

    def runs(self, problem: str | None = None, method: str | None = None) -> list[dict]:
        """Every record of this suite and setup (optionally of one problem or method), upgraded to the current schema."""
        base = self.root / "runs" / self.suite / self.setup
        pattern = f"{problem or '*'}/d*/{method or '*'}/tol*-s*.json"
        return [upgrade(r) for r in (read_json(p) for p in sorted(base.glob(pattern))) if r is not None]

    def migrate(self) -> int:
        """Rewrite every record of this suite and setup in the current schema; returns how many changed."""
        changed = 0
        for path in sorted((self.root / "runs" / self.suite / self.setup).glob("*/d*/*/tol*-s*.json")):
            record = read_json(path)
            if record is None:
                continue
            new = upgrade(record)
            if new is not record:
                key = self.key(new["problem"], new["d"], new["method"], new["tol"], new["seed"])
                write_json(path, {"key": key, **new, "migrated": now()})
                changed += 1
        return changed

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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="The benchmark's store of runs.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="rewrite every record of every suite and setup in the current schema")
    parser.parse_args(argv)
    for suite_dir in sorted((ROOT / "runs").glob("*")):
        for setup_dir in sorted(suite_dir.glob("*")):
            n = Store(suite_dir.name, setup_dir.name).migrate()
            print(f"{suite_dir.name}/{setup_dir.name}: {n} records migrated", flush=True)


if __name__ == "__main__":
    main()
