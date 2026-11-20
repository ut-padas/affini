"""
Plan, run and follow the manuscript study on one machine.

    python -m studies.manuscript.run plan   --machine studies/manuscript/machines/workstation.toml
    python -m studies.manuscript.run run    --machine ... [--experiments convergence,dimension] [--only REGEX]
    python -m studies.manuscript.run status --machine ...

Results go to ``<results>/cases/<key>.json`` (one per case, see case.py) and logs to
``<results>/logs/``. ``run`` goes through the cases in spec.cost_order, each in its own process with
the machine's device, timeout and MPI launcher, and resumes where it stopped: a finished or failed
case is not run again (``--retry-failed`` reruns failures). Before each case it projects the case's
time and host memory from the cases already measured (another seed of the same case, or the two
next-looser tolerances of its series), and skips it, recorded as ``skipped`` with the reason, when
the projection exceeds the machine's budget or a looser tolerance of its series already ran out of
memory or time; a later run on a larger machine tries skipped cases again.
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from studies.manuscript.spec import Case, cases, cost_order  # noqa: E402

# --smoke: the cases at this size, to check a machine end to end in minutes; its results go to their own folder.
SMOKE = {"n0": 3000, "k0": 8, "m": 40, "n_test": 5000, "tolerances_per_series": 2}

# How a case can end without finishing; the first four stop tighter tolerances of the same series.
EXHAUSTED = ("host_memory_guard", "device_out_of_memory", "timeout", "killed")


def load_machine(path: Path) -> dict:
    """A machine profile (machines/*.toml) with its defaults filled in."""
    m = tomllib.loads(path.read_text())
    import psutil

    m.setdefault("device", "auto")
    m.setdefault("distributed_device", "cpu")
    m.setdefault("mpi_launcher", ["mpiexec"])
    m.setdefault("max_ranks", os.cpu_count() or 1)
    m.setdefault("case_timeout_h", 4.0)
    m.setdefault("host_memory_gib", 0.9 * psutil.virtual_memory().total / 2**30)
    m.setdefault("exact_max_dim", None)
    m.setdefault("results", "studies/manuscript/output")
    return m


def result_path(results: Path, case: Case) -> Path:
    return results / "cases" / f"{case.key}.json"


def read_result(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def measured(results: Path) -> dict[str, dict]:
    """Finished results by case key: wall time and host peak, the projection's inputs."""
    out = {}
    for path in (results / "cases").glob("*.json"):
        r = read_result(path)
        if r and r.get("status") in ("done", "failed"):
            out[r["case"]["key"]] = r
    return out


def tolerance(case: Case) -> float:
    return case.tol if case.tol is not None else case.atol


def project(case: Case, by_key: dict[str, dict], all_cases: list[Case]) -> tuple[float, float] | None:
    """
    (seconds, host GiB) projected for ``case``: the largest measurement of the same case at another
    seed, else the geometric extrapolation from the two next-looser tolerances measured in its series
    (any seed), else None (unknown: the case runs).
    """
    def cost(r):
        return r["wall_s"], r["memory"]["host_peak_gib"]

    same = [by_key[c.key] for c in all_cases
            if c.key in by_key and c.series == case.series and tolerance(c) == tolerance(case) and c.seed != case.seed]  # fmt: skip
    done = [r for r in same if r["status"] == "done"]
    if done:
        return tuple(max(v) for v in zip(*map(cost, done), strict=True))
    looser = {}
    for c in all_cases:
        r = by_key.get(c.key)
        if r and r["status"] == "done" and c.series == case.series and tolerance(c) > tolerance(case):
            t = tolerance(c)
            looser[t] = tuple(max(a, b) for a, b in zip(looser.get(t, (0.0, 0.0)), cost(r), strict=True))
    if len(looser) < 2:
        return None
    (t2, c2), (t1, c1) = sorted(looser.items())[:2]  # t2 < t1: the two closest looser tolerances
    steps = math.log(t2 / tolerance(case)) / math.log(t1 / t2)
    return tuple(b * (b / a) ** steps if a > 0 else b for a, b in zip(c1, c2, strict=True))


def blocked(case: Case, by_key: dict[str, dict], all_cases: list[Case]) -> str | None:
    """Why a looser tolerance of the same series and seed makes this case pointless, or None."""
    for c in all_cases:
        r = by_key.get(c.key)
        if (r and r["status"] == "failed" and c.series == case.series and c.seed == case.seed
                and tolerance(c) > tolerance(case) and (r.get("failure") or {}).get("class") in EXHAUSTED):  # fmt: skip
            return f"the looser tolerance {tolerance(c):g} of its series ended with {r['failure']['class']}"
    return None


def smoke_cases(all_cases: list[Case]) -> list[Case]:
    """
    The cases at --smoke size (ranks capped at 2, iterative runs at 2 rounds), keeping the loosest
    tolerances of each series: enough for every code path and the projection. Tighter ones are left
    out because a cluster of 40 points cannot resolve them in d = 10, where the queues outgrow a
    96 GiB GPU and the host's memory.
    """
    import dataclasses

    small = {c.key: c for c in (dataclasses.replace(c, n0=SMOKE["n0"], k0=SMOKE["k0"], m=SMOKE["m"],
                                                     ranks=min(c.ranks, 2), rounds=min(c.rounds, 2))
                                for c in all_cases)}  # fmt: skip
    loosest: dict[tuple, list[float]] = {}
    for c in small.values():
        loosest.setdefault(c.series, []).append(tolerance(c))
    keep = {s: sorted(set(t), reverse=True)[: SMOKE["tolerances_per_series"]] for s, t in loosest.items()}
    return [c for c in small.values() if tolerance(c) in keep[c.series]]


def select(all_cases: list[Case], a) -> list[Case]:
    """The cases ``run`` / ``plan`` work on, in cost order, after the command line's filters and shard."""
    keep = [c for c in sorted(all_cases, key=cost_order)
            if (not a.experiments or set(c.experiments) & set(a.experiments.split(",")))
            and (a.only is None or re.search(a.only, c.key))
            and (a.seeds is None or c.seed in {int(s) for s in a.seeds.split(",")})]  # fmt: skip
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        keep = keep[i::n]
    return keep


def decide(case: Case, machine: dict, by_key: dict, all_cases: list[Case], force: bool) -> tuple[str, str, tuple | None]:
    """(state, reason, projection): state is done, failed, skip or run."""
    r = by_key.get(case.key)
    if r and r["status"] == "done":
        return "done", "", None
    if r and r["status"] == "failed":
        return "failed", (r.get("failure") or {}).get("class", ""), None
    if case.ranks > machine["max_ranks"]:
        return "skip", f"needs {case.ranks} ranks, the machine has {machine['max_ranks']}", None
    why = blocked(case, by_key, all_cases)
    if why and not force:
        return "skip", why, None
    proj = project(case, by_key, all_cases)
    if proj and not force:
        if proj[0] > 3600 * machine["case_timeout_h"]:
            return "skip", f"projected {proj[0] / 3600:.1f} h, over the {machine['case_timeout_h']:g} h limit", proj
        if proj[1] > machine["host_memory_gib"]:
            return "skip", f"projected {proj[1]:.0f} GiB host memory, over the {machine['host_memory_gib']:.0f} GiB budget", proj
    return "run", "", proj


def command(case: Case, machine: dict, out: Path) -> tuple[list[str], dict]:
    """
    The command and environment of one case: CPU ranks for the distributed experiment, the machine's
    device otherwise. Every distributed case, the single-rank one too, gets one thread per rank, so
    its times measure what ranks and work stealing buy and not the thread pools: with the defaults,
    one rank would use every core and P ranks would oversubscribe them.
    """
    cmd = [sys.executable, "-m", "studies.manuscript.case", "--case", json.dumps(case.as_dict()), "--out", str(out)]
    if machine["exact_max_dim"] is not None:
        cmd += ["--exact-max-dim", str(machine["exact_max_dim"])]
    if machine.get("smoke"):
        cmd += ["--n-test", str(SMOKE["n_test"])]
    cpu = "distributed" in case.experiments
    env = {**os.environ, "PYTHONPATH": str(ROOT), "SURROGATE_DEVICE": machine["distributed_device"] if cpu else machine["device"],
           "SURROGATE_MPI": "on" if case.ranks > 1 else "off"}  # fmt: skip
    if cpu:
        env |= {"OMP_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if case.ranks > 1:
        cmd = [*machine["mpi_launcher"], "-n", str(case.ranks), *cmd]
    return cmd, env


def execute(case: Case, machine: dict, results: Path, launch=subprocess.Popen) -> dict:
    """
    Run one case in its own process group and return its result. The case reads an empty stdin (a
    launcher that forwards stdin must not see anything), and a case that times out or is killed gets
    a result written here, since it could not write one itself.
    """
    out = result_path(results, case)
    log = results / "logs" / f"{case.key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd, env = command(case, machine, out)
    t0 = time.perf_counter()
    timeout = 3600 * machine["case_timeout_h"]
    with log.open("a") as fh:
        fh.write(f"### {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(cmd[:-4])} ...\n")
        fh.flush()
        proc = launch(cmd, env=env, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            klass = None
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code, klass = proc.returncode, "timeout"
        fh.write(f"### exit {code} after {time.perf_counter() - t0:.0f} s\n")
    r = read_result(out)
    if r and r.get("status") in ("done", "failed") and klass is None:
        return r
    if klass is None:
        klass = "killed" if code in (-signal.SIGKILL, 128 + signal.SIGKILL) else "crashed"
    r = {"schema": 1, "case": case.as_dict(), "status": "failed", "stage": (r or {}).get("stage", "unknown"),
         "failure": {"class": klass, "type": "exit", "message": f"exit code {code}; see {log.relative_to(results)}"},
         "wall_s": time.perf_counter() - t0, "memory": {"host_peak_gib": float("nan"), "device_peak_gib": float("nan")}}  # fmt: skip
    write_record(out, r)
    return r


def write_skip(case: Case, results: Path, reason: str, proj) -> None:
    write_record(result_path(results, case), {"schema": 1, "case": case.as_dict(), "status": "skipped", "reason": reason,
                                              "projection": {"wall_s": proj[0], "host_gib": proj[1]} if proj else None})  # fmt: skip


def write_record(out: Path, record: dict) -> None:
    """
    A result the runner writes itself (a case killed or skipped), atomically and with its time, as
    ``src.infra.records.write_json`` writes the cases' own. The runner never imports ``src``: on a GPU
    machine that opens a CUDA context, which the runner would hold for the whole study.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}-{os.getpid()}.tmp")
    tmp.write_text(json.dumps({**record, "recorded": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    tmp.replace(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "run", "status"))
    parser.add_argument("--machine", type=Path, required=True, help="a machine profile, studies/manuscript/machines/*.toml")
    parser.add_argument("--results", type=Path, default=None, help="default: the profile's results folder")
    parser.add_argument("--experiments", default=None, help="comma-separated: keep cases of these experiments")
    parser.add_argument("--only", default=None, help="regular expression on the case key")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds to keep")
    parser.add_argument("--shard", default=None, help="i/n: every n-th case from the i-th, for job arrays")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true", help="run cases the projection would skip")
    parser.add_argument("--json", action="store_true", help="plan and status as JSON lines")
    parser.add_argument("--smoke", action="store_true", help="the cases at a tiny size, into <results>-smoke")
    a = parser.parse_args(argv)
    machine = load_machine(a.machine)
    results = (a.results or ROOT / machine["results"]).resolve()
    all_cases = cases()
    if a.smoke:
        machine["smoke"] = True
        results = results.with_name(results.name + "-smoke") if a.results is None else results
        all_cases = smoke_cases(all_cases)
    chosen = select(all_cases, a)

    if a.command in ("plan", "status"):
        by_key = measured(results)
        states = Counter()
        for c in chosen:
            state, reason, proj = decide(c, machine, by_key, all_cases, a.force)
            if a.command == "status" and state == "run":
                saved = read_result(result_path(results, c))
                state, reason = ("skipped", saved.get("reason", "")) if saved and saved.get("status") == "skipped" else ("pending", "")
            states[state] += 1
            if a.json:
                print(json.dumps({"key": c.key, "experiments": c.experiments, "state": state, "reason": reason,
                                  "projected_s": proj[0] if proj else None, "projected_gib": proj[1] if proj else None}))  # fmt: skip
            elif state not in ("done",) and (a.command == "plan" or state != "pending"):
                eta = f"  ~{proj[0] / 60:.0f} min, {proj[1]:.0f} GiB" if proj else ""
                print(f"{state:8s} {c.key}{eta}{'  (' + reason + ')' if reason else ''}")
        print(f"# {len(chosen)} cases: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())), file=sys.stderr)
        return

    for c in chosen:
        by_key = measured(results)
        if a.retry_failed and c.key in by_key and by_key[c.key]["status"] == "failed":
            del by_key[c.key]
        state, reason, proj = decide(c, machine, by_key, all_cases, a.force)
        if state in ("done", "failed"):
            continue
        if state == "skip":
            write_skip(c, results, reason, proj)
            print(f"[{time.strftime('%H:%M:%S')}] skip {c.key}: {reason}", flush=True)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] run  {c.key}" + (f" (projected {proj[0] / 60:.0f} min)" if proj else ""), flush=True)
        r = execute(c, machine, results)
        tail = "" if r["status"] == "done" else f": {r['failure']['class']}"
        print(f"[{time.strftime('%H:%M:%S')}] {r['status']} {c.key} in {r['wall_s']:.0f} s{tail}", flush=True)


if __name__ == "__main__":
    main()
