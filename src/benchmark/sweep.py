"""
Sweep a suite's grid: run every (method, problem, d, tolerance, seed) not yet in the store, cheapest first.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python -m src.benchmark.sweep functions --d 2,3,5,10,20
    SURROGATE_DEVICE=cpu SURROGATE_MPI=off SURROGATE_ODE_WORKERS=12 PYTHONPATH=. .venv/bin/python -m src.benchmark.sweep step_maps

Problems come in order of input dimension (the function suite: every problem at each d in turn; the step-map
suite: its ladder, the families of ``--first`` before the rest). For each method, tolerances go from the
loosest to the tightest, since the cost only grows as the tolerance tightens. The first seed runs as its own
job, in its own process group, under ``--limit`` seconds of wall time; if it finishes, the other seeds run
in one more job. A job over the limit is killed with its whole group (so no rank is left waiting in a
collective) and recorded as failed, with the limit as the reason, and the method's tighter tolerances on
that problem are skipped; a method that fails a family's loosest tolerance (the same function at a smaller
d, or a smaller size of a step-map family) is not run on the family's larger sizes, which only cost more.
A job that crashes leaves no record: it is reported, and runs again next time. Recorded cases are not
rerun, unless ``--since`` names a time after which a record must have been written (records of older code). After each recorded case a background process redraws that problem's figures, one redraw at a time,
while the next case runs. Every job imports ``src`` afresh, so do not edit it while a sweep runs.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from src import PROJECT_DIR
from src.benchmark.store import Store

PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python")


@dataclass
class Plan:
    """What one suite's sweep runs: its problem groups in order, and how to run, judge and redraw a case."""

    store: Store
    groups: Iterable  # (problem, d, family) in run order
    methods: list
    tolerances: Callable  # (problem, d) -> tolerances, loosest first
    seeds: tuple
    command: Callable  # (problem, d, method, tol, seeds) -> argv of one job
    redraw: Callable  # (problem, d) -> argv of the problem's redraw
    skip: Callable = lambda method, problem, d: False  # noqa: E731
    since: str | None = None  # records written before this time (ISO, e.g. 2026-11-19T00:00) count as missing and are run again
    prepare: Callable = lambda problem: None  # noqa: E731  (work before a problem's cases, outside any limit)
    env: dict | None = None


def run_job(argv: list, limit: float, log, env: dict | None = None) -> bool:
    """One job in its own process group; False when it ran past ``limit`` seconds and was killed."""
    proc = subprocess.Popen(argv, cwd=PROJECT_DIR, env={**os.environ, **(env or {})}, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                            start_new_session=True)  # fmt: skip
    try:
        proc.wait(timeout=limit)
        return True
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return False


class Redrawer:
    """Background redraws, one at a time, so the last one started, which sees every record so far, is also the last to write."""

    def __init__(self, log):
        self.log, self.proc = log, None

    def start(self, argv: list) -> None:
        self.wait()
        self.proc = subprocess.Popen(argv, cwd=PROJECT_DIR, env={**os.environ, "SURROGATE_MPI": "off"}, stdin=subprocess.DEVNULL,
                                     stdout=self.log, stderr=self.log)  # fmt: skip

    def wait(self) -> None:
        if self.proc is not None:
            self.proc.wait()


def sweep(plan: Plan, limit: float) -> None:
    """Run the plan's missing cases (see the module docstring)."""
    store, stuck = plan.store, set()  # stuck: (method, family) whose loosest tolerance failed

    def done(problem, d, method, tol, seed) -> bool:
        record = store.read(problem, d, method, tol, seed)
        return record is not None and (plan.since is None or record.get("recorded", "") >= plan.since)

    with open(store.log, "a") as log:
        redrawer = Redrawer(log)
        for problem, d, family in plan.groups:
            plan.prepare(problem)
            for method in plan.methods:
                if plan.skip(method, problem, d) or (method, family) in stuck:
                    continue
                for i, tol in enumerate(plan.tolerances(problem, d)):
                    first = plan.seeds[0]
                    if not done(problem, d, method, tol, first):
                        t0 = time.time()
                        stale = ["--force"] if store.exists(problem, d, method, tol, first) else []  # an older record: overwrite it
                        ok = run_job(plan.command(problem, d, method, tol, [first]) + stale, limit, log, plan.env)
                        if not ok:
                            store.write({"problem": problem, "d": d, "method": method, "tol": tol, "seed": first,
                                         "failed": f"over the {limit:g} s time limit"})  # fmt: skip
                        elif not done(problem, d, method, tol, first):
                            print(f"{problem} d={d} {method} {tol:g}: crashed (see {store.log})", flush=True)
                            break
                        print(f"{problem} d={d} {method} {tol:g}: {'done' if ok else 'over the limit'} ({time.time() - t0:.0f} s)", flush=True)
                        redrawer.start(plan.redraw(problem, d))
                    if "failed" in store.read(problem, d, method, tol, first):
                        if i == 0:
                            stuck.add((method, family))
                        break  # tighter tolerances only cost more
                    rest = [s for s in plan.seeds[1:] if not done(problem, d, method, tol, s)]
                    if rest:
                        stale = ["--force"] if any(store.exists(problem, d, method, tol, s) for s in rest) else []
                        ok = run_job(plan.command(problem, d, method, tol, rest) + stale, len(rest) * limit, log, plan.env)
                        for s in rest:
                            if not ok and not done(problem, d, method, tol, s):
                                store.write({"problem": problem, "d": d, "method": method, "tol": tol, "seed": s,
                                             "failed": f"over the {limit:g} s time limit"})  # fmt: skip
                        redrawer.start(plan.redraw(problem, d))
        redrawer.wait()


def functions_plan(a) -> Plan:
    """The function suite on ``a.ranks`` MPI ranks (AFINNI and ART train collectively; the schemes spread their oracle calls)."""
    from src.benchmark import functions, methods
    from src.problems import FUNCTIONS, VECTOR_FUNCTIONS

    methods.SETUP = f"cpu-{a.ranks}rank"
    fast = ("rosenbrock", "genz_oscillatory", "genz_corner_peak", "genz_product_peak", "genz_gaussian", "lotka_volterra")
    if a.problems:
        problems = a.problems.split(",")
    elif a.vector:
        problems = list(VECTOR_FUNCTIONS)
    else:
        problems = list(fast) + [p for p, prob in FUNCTIONS.items() if p not in fast and not prob.vector]
    mpiexec = str(PROJECT_DIR / ".venv" / "bin" / "mpiexec")

    def command(problem, d, method, tol, seeds):
        return [mpiexec, "-n", str(a.ranks), PYTHON, "-m", "src.benchmark.functions", "--problems", problem, "--d", str(d), "--methods", method,
                "--tols", f"{tol:g}", "--seeds", ",".join(map(str, seeds))]  # fmt: skip

    def redraw(problem, d):
        return [PYTHON, "-m", "src.benchmark.functions", "--replot", "--setup", methods.SETUP, "--d", str(d), "--problems", problem]

    return Plan(functions.store(), [(p, d, p) for d in map(int, a.d.split(",")) for p in problems], a.methods.split(","),
                lambda problem, d: sorted(functions.tolerances(d), reverse=True)[: 1 if a.loosest else None], functions.SEEDS, command, redraw,
                skip=lambda method, problem, d: method == "afinni" and d >= 10,  # AFINNI from 10-D: tens of GB of host memory per rank
                env={"SURROGATE_DEVICE": "cpu", "SURROGATE_MPI": "on", "OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                     "PYTHONPATH": ".", "PATH": f"{PROJECT_DIR / '.venv' / 'bin'}:{os.environ['PATH']}"})  # fmt: skip


def step_maps_plan(a) -> Plan:
    """The step-map suite, serially: each case one process (the solver's own pool serves the oracle)."""
    from src.benchmark import step_maps as sm

    def groups():
        for name in sm.ladder(a.max_d, tuple(filter(None, a.first.split(",")))):
            p = sm.STEP_MAPS[name]
            if sm.dim(name) >= a.min_d and p.regime in a.regimes.split(","):
                yield name, sm.dim(name), sm.family(name)

    def prepare(name):
        t0 = time.time()
        sm.data(name)
        sm.solver_cost(name)
        sm.dynamics(name)
        print(f"{name}: trajectories and solver cost ready ({time.time() - t0:.0f} s)", flush=True)

    def command(name, d, method, tol, seeds):
        return [sys.executable, "-m", "src.benchmark.step_maps", "--problems", name, "--methods", method, "--tols", f"{tol:g}"]

    return Plan(sm.STORE, groups(), a.methods.split(","),
                lambda name, d: sorted(sm.STEP_TOLERANCES, reverse=True)[: 1 if a.loosest else None], (1,), command,
                lambda name, d: [sys.executable, "-m", "src.benchmark.step_maps", "--problems", name, "--redraw"],
                skip=lambda method, name, d: method == "afinni" and d >= 10, prepare=prepare)  # fmt: skip


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="suite", required=True)
    f = sub.add_parser("functions", help="the test functions of src.problems")
    f.add_argument("--d", default="2,3,5,10,20")
    f.add_argument("--problems", default=None, help="default: a fast set first, then every other scalar problem")
    f.add_argument("--vector", action="store_true", help="the vector-valued problems instead of the scalar ones")
    f.add_argument("--methods", default="chebtt,chebtt_metric,art_quad,art,afinni")
    f.add_argument("--ranks", type=int, default=8)
    f.add_argument("--limit", type=float, default=90.0, help="wall seconds per job, start-up included")
    s = sub.add_parser("step_maps", help="the ODE step maps of src.problems.ode")
    s.add_argument("--methods", default="chebtt_roll,chebtt_box,chebtt_smooth,chebtt_adaptive,slowtable,art_quad,art,afinni")
    s.add_argument("--max-d", type=int, default=80)
    s.add_argument("--min-d", type=int, default=1)
    s.add_argument("--first", default="", help="families (comma-separated) to run before the rest")
    s.add_argument("--regimes", default="stiff,nonstiff,chaotic,longstep", help="the regimes (comma-separated) whose problems run")
    s.add_argument("--limit", type=float, default=240.0, help="wall seconds per job")
    for p in (f, s):
        p.add_argument("--loosest", action="store_true", help="only each problem's loosest tolerance")
        p.add_argument("--since", default=None, help="run again (overwriting) every record written before this time, e.g. 2026-11-19T00:00")
    a = parser.parse_args(argv)
    plan = functions_plan(a) if a.suite == "functions" else step_maps_plan(a)
    plan.since = a.since
    sweep(plan, a.limit)


if __name__ == "__main__":
    main()
