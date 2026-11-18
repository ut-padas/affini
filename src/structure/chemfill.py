"""
Climb the stiff-system ladder: every (method, problem, tolerance) of the stiff bench whose runs train in
reasonable time, smallest systems first.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off SURROGATE_CHEM_WORKERS=12 PYTHONPATH=. \\
        .venv/bin/python -m src.structure.chemfill --limit 240

Problems go in order of input dimension: each family of ``chemistry.LADDER`` at every size, and the
fixed-size test problems (Robertson, Akzo Nobel, HIRES, POLLU) at theirs. A problem's trajectories, the
solver's cost per step and its dynamics diagnostics are computed first, outside the time limit. Each case then runs as its own
process under ``--limit`` seconds of wall time, tolerances from the loosest; one over the limit is
recorded as failed (``failed``: the limit) and the tighter tolerances are skipped. A method that cannot
finish a family's loosest tolerance in time is not run on that family's larger sizes: they only cost
more. A crashed case leaves no record and runs again next time; recorded cases are never rerun. After
each problem the bench redraws its error plot, the summary and the ladder overview.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.structure import chembench

FIXED = {"robertson": "robertson", "chemakzo": "chemakzo", "hires": "hires", "primordial": "primordial", "pollu": "pollu"}  # problem: family


def family(name: str) -> str:
    return FIXED.get(name, name.rsplit("_d", 1)[0])


def ladder(max_d: int) -> list[str]:
    """Problems of the ladder up to ``max_d`` inputs, by input dimension (ties: by name)."""
    names = [n for n in chembench.PROBLEMS if n in FIXED or family(n) != n]
    dims = {n: len(chembench.PROBLEMS[n].net.species) for n in names}
    return sorted((n for n in names if dims[n] <= max_d), key=lambda n: (dims[n], n))


def _case(args: list[str], limit: float, log) -> bool:
    """One bench case in its own process group; False when it ran past ``limit`` seconds and was killed."""
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen([sys.executable, "-m", "src.structure.chembench", *args], cwd=root, stdin=subprocess.DEVNULL,
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
    parser.add_argument("--limit", type=float, default=240.0, help="wall seconds per case")
    a = parser.parse_args(argv)
    stuck: set[tuple[str, str]] = set()  # (method, family) whose loosest tolerance ran out of time
    log_path = chembench.OUT / "chemfill.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        for name in ladder(a.max_d):
            p = chembench.PROBLEMS[name]
            t0 = time.time()
            chembench.data(name)
            chembench.solver_cost(name)
            chembench.dynamics(name)
            print(f"{name}: trajectories and solver cost ready ({time.time() - t0:.0f} s)", flush=True)
            ran = False
            for method in a.methods.split(","):
                if (method, family(name)) in stuck or (method == "afinni" and len(p.net.species) >= 10):
                    continue  # AFINNI from 10 inputs: tens of GB of host memory (the box bench's rule)
                for i, tol in enumerate(sorted(p.tolerances, reverse=True)):
                    path = chembench.CACHE / name / f"{method}_tol{tol:g}.json"
                    if not path.exists():
                        t0 = time.time()
                        ok = _case(["--problems", name, "--methods", method, "--tols", f"{tol:g}"], a.limit, log)
                        ran = True
                        if not ok:
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_text(json.dumps({"method": method, "problem": name, "tol": tol,
                                                        "failed": f"over the {a.limit:g} s time limit"}))  # fmt: skip
                        elif not path.exists():
                            print(f"{name} {method} {tol:g}: crashed (see {log_path})", flush=True)
                            break
                        print(f"{name} {method} {tol:g}: {'done' if ok else 'over the limit'} ({time.time() - t0:.0f} s)", flush=True)
                    if "failed" in json.loads(path.read_text()):
                        if i == 0:
                            stuck.add((method, family(name)))
                        break
            if ran:
                chembench.plot(name, list(chembench.METHODS))
                chembench.write_summary()
                chembench.plot_ladder()
                chembench.plot_overview()
                chembench.plot_speedup()


if __name__ == "__main__":
    main()
