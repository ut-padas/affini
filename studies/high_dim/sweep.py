"""
The run matrix of the high-dimensional study: for every problem and variant, d rises until a
run fails for a reason a larger d cannot fix (out of memory, time limit, crash), and at every
d the tolerance falls from coarse to fine until a run does not reach it.

Each run is its own process (``run_one.py``) with a hard time limit, so an out-of-memory
error, a kernel OOM kill or a hang costs one run, not the sweep. Results go to
``<out>/runs/<id>.json`` (one record per run) with the run's log beside it; a rerun of the
same command skips every run that has a final record, so the sweep can be stopped and
resumed at any time. ``collect.py`` turns the records into tables and plots.

"Reached" means the run finished within its limits with every node accepted (``converged``:
no AFINNI cluster unresolved at ``max_levels``, no ART fallback leaf). The finest tolerance a
method reaches at d is the last reached tolerance of the ladder.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from studies.high_dim.problems import PROBLEMS
from studies.high_dim.settings import DEFAULT_VARIANTS, VARIANTS

REPO = Path(__file__).resolve().parents[2]
TOLS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
# Where the leaf model represents the function exactly, the ladder continues to find the
# implementation's floor (the affine fit's ridge bias grows with d; see TODO.md).
EXACT_TOLS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
DIMS = (20, 40, 60, 80, 100, 120, 160, 200, 250, 300)
# Failures that a larger d cannot fix: the sweep stops raising d for that (problem, variant).
STOP_D = {"gpu_oom", "host_oom", "killed", "time_limit", "crash", "error"}
GRACE = 120.0  # seconds beyond the run's own time limit before the sweep kills it


def run_id(problem: str, variant: str, d: int, tol: float) -> str:
    return f"{problem}__{variant}__d{d:03d}__tol{tol:.0e}"


def ladder(problem: str, variant: str, tols, exact_tols) -> list[float]:
    exact = VARIANTS[variant].leaf in PROBLEMS[problem].exact_for
    return list(tols) + (list(exact_tols) if exact else [])


def launch(args, problem: str, variant: str, d: int, tol: float, path: Path) -> dict:
    """Run one ``run_one`` process; return its record, completed with the failure if the process died."""
    log = path.with_suffix(".log")
    cmd = [
        sys.executable, "-m", "studies.high_dim.run_one", "--variant", variant, "--problem", problem,
        "--d", str(d), "--tol", repr(tol), "--out", str(path), "--n-test", str(args.n_test),
        "--time-limit", str(args.time_limit), "--seed", str(args.seed),
    ]  # fmt: skip
    if args.library_min_pts:
        cmd.append("--library-min-pts")
    t = time.time()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=args.time_limit + GRACE)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = "timeout"
    record = json.loads(path.read_text()) if path.exists() else {}
    if record.get("status") in (None, "running"):
        # The process died before writing its final record.
        if code == "timeout":
            failure = "time_limit"
        elif isinstance(code, int) and code == -signal.SIGKILL:
            failure = "killed"  # most often the kernel's OOM killer (host RAM); see dmesg
        elif isinstance(code, int) and code < 0:
            failure = "crash"
        else:
            failure = "error"
        record.update(status="failed", reached=False, failure=failure, return_code=str(code),
                      problem=problem, variant=variant, d=d, tol=tol)  # fmt: skip
        record["failure_message"] = record.get("failure_message") or _tail(log)
    record["wall_process"] = time.time() - t
    path.write_text(json.dumps(record, indent=1))
    return record


def _tail(log: Path, n: int = 3) -> str:
    try:
        return " | ".join(log.read_text(errors="replace").strip().splitlines()[-n:])[-600:]
    except OSError:
        return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO / "studies/high_dim/results"))
    parser.add_argument("--problems", default=",".join(PROBLEMS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--dims", default=",".join(map(str, DIMS)))
    parser.add_argument("--tols", default=",".join(map(repr, TOLS)))
    parser.add_argument("--exact-tols", default=",".join(map(repr, EXACT_TOLS)))
    parser.add_argument("--time-limit", type=float, default=1800.0, help="seconds per run")
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--library-min-pts", action="store_true", help="min_pts_per_cluster = 100 at every d")
    parser.add_argument("--retry-failed", action="store_true", help="rerun records that failed with error/crash/killed")
    parser.add_argument("--dry-run", action="store_true", help="print the matrix without running")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny CPU-sized matrix (d = 4, 8; two tolerances; 90 s per run) to check the pipeline")  # fmt: skip
    args = parser.parse_args(argv)
    if args.smoke:
        args.dims, args.tols, args.exact_tols, args.time_limit, args.n_test = "4,8", "1.0,0.3", "1e-6", 90.0, 2000
    for var in ("AFINNI_DEVICE", "AFINNI_MPI"):
        if var not in os.environ:
            print(f"Set {var} explicitly (AGENTS.md), e.g. AFINNI_DEVICE=gpu AFINNI_MPI=off.", file=sys.stderr)
            return 2

    problems, variants = args.problems.split(","), args.variants.split(",")
    for p in problems:
        if p not in PROBLEMS:
            parser.error(f"unknown problem {p!r}; choose from {sorted(PROBLEMS)}")
    for v in variants:
        if v not in VARIANTS:
            parser.error(f"unknown variant {v!r}; choose from {sorted(VARIANTS)}")
    dims = [int(x) for x in args.dims.split(",")]
    tols = [float(x) for x in args.tols.split(",")]
    exact_tols = [float(x) for x in args.exact_tols.split(",") if x]
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    manifest = vars(args) | {
        "device": os.environ["AFINNI_DEVICE"], "mpi": os.environ["AFINNI_MPI"], "started": time.time(),
    }  # fmt: skip
    (out / f"manifest_{int(time.time())}.json").write_text(json.dumps(manifest, indent=1))

    stopped: dict[tuple[str, str], str] = {}
    for d in dims:
        for problem in problems:
            for variant in variants:
                key = (problem, variant)
                if key in stopped:
                    continue
                for i, tol in enumerate(ladder(problem, variant, tols, exact_tols)):
                    rid = run_id(problem, variant, d, tol)
                    path = out / "runs" / f"{rid}.json"
                    record = json.loads(path.read_text()) if path.exists() else None
                    final = record is not None and record.get("status") in ("ok", "failed")
                    retry = final and args.retry_failed and record.get("failure") in ("error", "crash", "killed")
                    if args.dry_run:
                        print(f"{'done ' if final else 'todo '} {rid}")
                        continue
                    if not final or retry:
                        print(f"[{time.strftime('%H:%M:%S')}] {rid} ...", flush=True)
                        record = launch(args, problem, variant, d, tol, path)
                        print(f"    {record.get('status')} reached={record.get('reached')} failure={record.get('failure')} "
                              f"leaves={record.get('leaves')} t={record.get('time_train')}", flush=True)  # fmt: skip
                    if not record.get("reached"):
                        if i == 0 and record.get("failure") in STOP_D:
                            stopped[key] = f"d={d}: {record.get('failure')}"
                        break
    if stopped:
        print("Stopped raising d:", json.dumps({f"{p}/{v}": r for (p, v), r in stopped.items()}, indent=1))
    (out / "stopped.json").write_text(json.dumps({f"{p}/{v}": r for (p, v), r in stopped.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
