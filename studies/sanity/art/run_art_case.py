"""
Run one side of ``scripts/art_comparison.py`` (the reference ART or this library's ART) and save its
result as JSON.

Usage (from the repository root, PYTHONPATH=.)::

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off python studies/sanity/art/run_art_case.py \\
        --side reference|library --art-path ../Adaptive-Regression-Trees-ART- --out-json R.json \\
        -- --benchmark rosenbrock --dimension 2 --offset 1 --error-tolerance 0.05 --max-depth 20

Everything after ``--`` goes to art_comparison.py unchanged (its own flags and the reference's
benchmark flags), so both sides of a case see the same target, domain, test set and settings.
The comparison's summary is extended with the reference evaluator's RMSE, maximum absolute
error and full pointwise-relative quantiles; the JSON also records the wall time, rank 0's peak
resident memory, the device and the MPI size. This library's side may run under mpiexec; rank 0 writes.
An existing ``--out-json`` is not rerun.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class _Tee(io.TextIOBase):
    """Write to the real stdout and keep a copy, so the run's output still reaches the log."""

    def __init__(self, stream):
        self.stream, self.buffer_ = stream, io.StringIO()

    def write(self, s):
        self.stream.write(s)
        self.buffer_.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()


def load_art_comparison():
    spec = importlib.util.spec_from_file_location("art_comparison", ROOT / "scripts" / "art_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._summary

    def extended(evaluation, *args, **kwargs):
        out = original(evaluation, *args, **kwargs)
        m = evaluation.metrics
        out.update({
            "test_rmse": m["test_rmse"], "test_mse": m["test_mse"], "test_max_absolute_error": m["test_max_absolute_error"],
            "pointwise_relative_quantiles": {str(q): float(v) for q, v in evaluation.pointwise_relative_quantiles.items()},
            "oracle_min": float(evaluation.oracle_min), "oracle_max": float(evaluation.oracle_max),
        })  # fmt: skip
        return out

    module._summary = extended  # main() looks _summary up at call time
    return module


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: run_art_case.py --side reference|library --art-path P --out-json F -- <art_comparison args>")
    split = argv.index("--")
    own = argparse.ArgumentParser()
    own.add_argument("--side", choices=("reference", "library"), required=True)
    own.add_argument("--art-path", type=Path, required=True)
    own.add_argument("--out-json", type=Path, required=True)
    own.add_argument("--case", default=None)
    mine = own.parse_args(argv[:split])
    if mine.out_json.exists():
        print(f"{mine.out_json} exists; skipping")
        return

    module = load_art_comparison()
    from src.infra.backend import MPI

    rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    skip = "--skip-library" if mine.side == "reference" else "--skip-reference"
    sys.argv = ["art_comparison.py", "--art-path", str(mine.art_path), skip, *argv[split + 1 :]]
    tee = _Tee(sys.stdout)
    t = time.perf_counter()
    with contextlib.redirect_stdout(tee):
        module.main()
    wall = time.perf_counter() - t
    if rank != 0:
        return
    compare = [json.loads(line[len("COMPARE "):]) for line in tee.buffer_.getvalue().splitlines() if line.startswith("COMPARE ")]
    config = [json.loads(line[len("CONFIG "):]) for line in tee.buffer_.getvalue().splitlines() if line.startswith("CONFIG ")]
    if len(compare) != 1:
        raise SystemExit(f"expected one COMPARE line, got {len(compare)}")
    from src import IS_CUPY_BACKEND

    record = {
        "case": mine.case, "side": mine.side, "argv": argv[split + 1 :], "result": compare[0],
        "config": config[0] if config else None, "wall_s": wall, "mpi_size": size,
        "library_gpu": bool(IS_CUPY_BACKEND), "peak_rss_gib_rank0": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "env": {k: os.environ.get(k) for k in ("SURROGATE_DEVICE", "SURROGATE_MPI", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }  # fmt: skip
    mine.out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = mine.out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str))
    tmp.replace(mine.out_json)


if __name__ == "__main__":
    main()
