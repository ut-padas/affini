"""
Climb the ODE ladder: every (method, problem, tolerance) of the ODE bench whose runs train in
reasonable time, smallest systems first.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off SURROGATE_ODE_WORKERS=12 PYTHONPATH=. \\
        .venv/bin/python -m src.structure.ode.fill --limit 240

Problems go in order of input dimension (the families of ``--first`` before the rest): each family of ``ode.FAMILIES`` at every size, and the
fixed-size test problems (Robertson, Akzo Nobel, HIRES, POLLU) at theirs. A problem's trajectories, the
solver's cost per step and its dynamics diagnostics are computed first, outside the time limit. Each case then runs as its own
process under ``--limit`` seconds of wall time, tolerances from the loosest; one over the limit is
recorded as failed (``failed``: the limit) and the tighter tolerances are skipped. A method that cannot
finish a family's loosest tolerance in time is not run on that family's larger sizes: they only cost
more. A crashed case leaves no record and runs again next time; recorded cases are never rerun. After
each recorded case a background process redraws the problem's error plot, the summary, the ladder and
every regime's overview and speed plot (``bench.redraw``), while the next case runs.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.structure.ode import bench

FIXED = {"robertson": "robertson", "chemakzo": "chemakzo", "hires": "hires", "primordial": "primordial", "pollu": "pollu"}  # problem: family


def family(name: str) -> str:
    return FIXED.get(name, name.rsplit("_d", 1)[0])


def ladder(max_d: int, first: tuple[str, ...] = ()) -> list[str]:
    """Problems of the ladder up to ``max_d`` inputs, the families in ``first`` before the rest, each part by input dimension (ties: by name)."""
    names = [n for n in bench.PROBLEMS if n in FIXED or family(n) != n]
    dims = {n: len(bench.PROBLEMS[n].system.components) for n in names}
    return sorted((n for n in names if dims[n] <= max_d), key=lambda n: (family(n) not in first, dims[n], n))


ROOT = Path(__file__).resolve().parents[3]  # the repository


def _redraw(name: str, previous: subprocess.Popen | None, log) -> subprocess.Popen:
    """
    Start ``bench.redraw(name)`` in the background, once the redraw still running (``previous``) has ended:
    one at a time, so the last one started, which has every record so far, is also the last to write.
    """
    if previous is not None:
        previous.wait()
    return subprocess.Popen([sys.executable, "-m", "src.structure.ode.bench", "--problems", name, "--redraw"], cwd=ROOT,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log)  # fmt: skip


def _case(args: list[str], limit: float, log) -> bool:
    """One bench case in its own process group; False when it ran past ``limit`` seconds and was killed."""
    proc = subprocess.Popen([sys.executable, "-m", "src.structure.ode.bench", *args], cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=log, start_new_session=True)  # fmt: skip
    try:
        proc.wait(timeout=limit)
        return True
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return False


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--methods", default="chebtt_roll,chebtt_box,chebtt_smooth,chebtt_adaptive,slowtable,art_quad,art,afinni")
    parser.add_argument("--max-d", type=int, default=80)
    parser.add_argument("--first", default="", help="families (comma-separated) to run before the rest")
    parser.add_argument("--limit", type=float, default=240.0, help="wall seconds per case")
    a = parser.parse_args(argv)
    stuck: set[tuple[str, str]] = set()  # (method, family) whose loosest tolerance ran out of time
    log_path = bench.OUT / "odefill.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    drawer = None
    with open(log_path, "a") as log:
        for name in ladder(a.max_d, tuple(filter(None, a.first.split(",")))):
            p = bench.PROBLEMS[name]
            t0 = time.time()
            bench.data(name)
            bench.solver_cost(name)
            bench.dynamics(name)
            print(f"{name}: trajectories and solver cost ready ({time.time() - t0:.0f} s)", flush=True)
            for method in a.methods.split(","):
                if (method, family(name)) in stuck or (method == "afinni" and len(p.system.components) >= 10):
                    continue  # AFINNI from 10 inputs: tens of GB of host memory (the box bench's rule)
                for i, tol in enumerate(sorted(p.tolerances, reverse=True)):
                    path = bench.CACHE / name / f"{method}_tol{tol:g}.json"
                    if not path.exists():
                        t0 = time.time()
                        ok = _case(["--problems", name, "--methods", method, "--tols", f"{tol:g}"], a.limit, log)
                        if not ok:
                            bench.write_record(path, {"method": method, "problem": name, "tol": tol,
                                                      "failed": f"over the {a.limit:g} s time limit"})  # fmt: skip
                        elif not path.exists():
                            print(f"{name} {method} {tol:g}: crashed (see {log_path})", flush=True)
                            break
                        print(f"{name} {method} {tol:g}: {'done' if ok else 'over the limit'} ({time.time() - t0:.0f} s)", flush=True)
                        drawer = _redraw(name, drawer, log)
                    if "failed" in json.loads(path.read_text()):
                        if i == 0:
                            stuck.add((method, family(name)))
                        break
    if drawer is not None:
        drawer.wait()


if __name__ == "__main__":
    main()
