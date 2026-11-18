"""
Fill the bench's grid: run every (method, problem, d, tolerance) whose runs train in reasonable time.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python -m src.structure.fill --d 2,3,5,10,20

Cases go from the loosest tolerance to the tightest, since the cost only grows as the tolerance
tightens. Seed 1 runs first, as its own 8-rank job under ``--limit`` seconds of wall time; if it
finishes, the other seeds run. If it does not, the job is killed (its whole process group, so no
rank is left waiting in a collective), the case is cached as failed with the time limit as the
reason (the overview hatches it like a case over its evaluation budget), and the tighter tolerances
of that method and problem are skipped. Cached cases are never rerun. After each problem the bench
redraws that problem's plots, the summary and the overview.

AFINNI is not run at d >= 10: it needs tens of GB of host memory there (45 GB over 8 ranks on the
10-D wave, one rank 15 GB), which a time limit does not catch before the machine runs out of memory.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.structure import bench
from src.structure.problems import PROBLEMS

FAST = ("rosenbrock", "genz_oscillatory", "genz_corner_peak", "genz_product_peak", "genz_gaussian", "lotka_volterra")
ENV = {"SURROGATE_DEVICE": "cpu", "SURROGATE_MPI": "on", "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": "."}  # fmt: skip


def allowed(method: str, d: int) -> bool:
    return not (method == "afinni" and d >= 10)


def _bench(args: list[str], limit: float, ranks: int, log) -> bool:
    """One bench job; False when it ran past ``limit`` seconds and was killed with its whole process group."""
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, **ENV, "PATH": f"{root / '.venv' / 'bin'}:{os.environ['PATH']}"}
    cmd = [str(root / ".venv/bin/mpiexec"), "-n", str(ranks), str(root / ".venv/bin/python"), "-m", "src.structure.bench", *args]
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    try:
        proc.wait(timeout=limit)
        return True
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return False


def _record_time_limit(method, problem, d, tol, seed, limit, ranks) -> None:
    path = bench.cache_path(method, problem, d, tol, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = {"method": method, "problem": problem, "d": d, "tol": tol, "seed": seed, "failed": f"over the {limit:g} s time limit",
         "ranks": ranks, "device": "cpu"}  # fmt: skip
    np.savez_compressed(path, metrics=json.dumps(m))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--d", default="2,3,5,10,20")
    parser.add_argument("--problems", default=None, help="default: the fast set first, then every other problem")
    parser.add_argument("--methods", default="chebtt,art_quad,art,afinni,chebtree,sparsecheb,kernel")
    parser.add_argument("--limit", type=float, default=90.0, help="wall seconds per run, start-up included")
    parser.add_argument("--ranks", type=int, default=8)
    a = parser.parse_args(argv)
    import src.structure.schemes  # noqa: F401  (registers the schemes, so bench.METHODS knows them)

    bench.SETUP = f"cpu-{a.ranks}rank"
    problems = a.problems.split(",") if a.problems else list(FAST) + [p for p in PROBLEMS if p not in FAST]
    log_path = bench.OUT / bench.SETUP / "fill.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        for d in map(int, a.d.split(",")):
            for problem in problems:
                tols = sorted(PROBLEMS[problem].tolerances(d), reverse=True)
                ran = False
                for method in a.methods.split(","):
                    if not allowed(method, d):
                        continue
                    for tol in tols:
                        first = bench.cache_path(method, problem, d, tol, bench.SEEDS[0])
                        if not first.exists():
                            t0 = time.time()
                            ok = _bench(["--problems", problem, "--d", str(d), "--methods", method, "--tols", f"{tol:g}",
                                         "--seeds", str(bench.SEEDS[0])], a.limit, a.ranks, log)  # fmt: skip
                            ran = True
                            if not ok:
                                _record_time_limit(method, problem, d, tol, bench.SEEDS[0], a.limit, a.ranks)
                            print(f"{d}-D {problem} {method} {tol:g}: seed 1 {'done' if ok else 'over the limit'} "
                                  f"({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
                        m = json.loads(str(np.load(first)["metrics"]))
                        if "failed" in m:
                            break  # tighter tolerances only cost more
                        rest = [s for s in bench.SEEDS[1:] if not bench.cache_path(method, problem, d, tol, s).exists()]
                        if rest:
                            ok = _bench(["--problems", problem, "--d", str(d), "--methods", method, "--tols", f"{tol:g}",
                                         "--seeds", ",".join(map(str, rest))], len(rest) * a.limit, a.ranks, log)  # fmt: skip
                            ran = True
                            if not ok:
                                for s in rest:
                                    if not bench.cache_path(method, problem, d, tol, s).exists():
                                        _record_time_limit(method, problem, d, tol, s, a.limit, a.ranks)
                if ran:
                    subprocess.run([sys.executable, "-m", "src.structure.bench", "--replot", "--setup", bench.SETUP, "--d", str(d),
                                    "--problems", problem], cwd=Path(__file__).resolve().parents[2],
                                   env={**os.environ, "SURROGATE_MPI": "off"}, stdout=log, stderr=log, check=False)  # fmt: skip


if __name__ == "__main__":
    main()
