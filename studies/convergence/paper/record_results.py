"""
Record the manuscript's numbers as the current code measures them, with everything needed to
reproduce them: the commit, the machine, the libraries.

Usage: ``record_results.py --out OUT --machine "GH200 node"`` after ``collect_paper.py --out OUT``
(``run_manuscript.sh`` runs both). Writes ``studies/convergence/results/<date>_<commit>_<machine>/``:

- ``manuscript_results.csv``: one row per number the manuscript reports (table, row, problem, d,
  tolerance, quantity): the value measured here (seed 0), the seed replicates' range, and the
  paper's value beside it for reference;
- ``slopes.csv``, ``checks.csv``, ``cost.csv``: the collector's tables, copied;
- ``provenance.json``: the commit (and whether the tree had changes), the commits the runs started
  at, the machine (CPU, cores, RAM, GPU, driver, CUDA runtime), Python and library versions, the
  hash of ``uv.lock``, and the run-time settings;
- ``RESULTS.md``: the provenance and the numbers, one table per manuscript table.

It also adds the record to ``studies/convergence/results/INDEX.md``. The machine is described only
by ``--machine`` and the hardware it reports; no host name is written.
"""

import argparse
import contextlib
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "studies" / "convergence" / "results"
QUANTITIES = ("N", "N_c", "p_o", "e_in", "e", "e_nh", "p_in_good")
LIBRARIES = ("numpy", "scipy", "numba", "llvmlite", "cupy-cuda12x", "cupy", "mpi4py", "jax", "jaxlib",
             "psutil", "matplotlib", "knndbscan", "nvidia-ml-py")  # fmt: skip


def run(cmd: list[str], timeout: int = 30) -> str:
    """A command's stripped output, or "" when it is missing or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git(*args: str) -> str:
    return run(["git", "-C", str(ROOT), *args])


def cpu_model() -> str:
    for line in run(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def gpus() -> list[dict]:
    rows = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    out = []
    for line in rows.splitlines():
        name, memory, driver = (part.strip() for part in line.split(","))
        out.append({"name": name, "memory": memory, "driver": driver})
    return out


def cuda_runtime() -> str:
    """The CUDA runtime CuPy uses, as major.minor, or "" without CuPy or a GPU."""
    code = "import cupy; v = cupy.cuda.runtime.runtimeGetVersion(); print(f'{v // 1000}.{v % 1000 // 10}')"
    return run([sys.executable, "-c", code])


def mpi_library() -> str:
    code = "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"
    return " ".join(run([sys.executable, "-c", code]).split())


def run_settings(out: Path) -> dict:
    """The commits and settings each ``run_paper.sh`` invocation recorded when it started."""
    text = (out / "provenance.txt").read_text() if (out / "provenance.txt").exists() else ""
    # Output directories written by checkouts of the study older than the "repo" keys and the
    # SURROGATE_ variables name them "afinni" and AFINNI_; both are read, so such runs record too.
    commits = sorted(set(re.findall(r"^(?:repo|afinni): ([0-9a-f]{40})", text, re.M)))
    dirty = sorted({int(n) for n in re.findall(r"^(?:repo|afinni)_dirty: (\d+)", text, re.M)})
    env = sorted(set(re.findall(r"^((?:SURROGATE|AFINNI)_DEVICE=\S* (?:SURROGATE|AFINNI)_MPI=\S*)", text, re.M)))
    starts = re.findall(r"^date: (\S+)", text, re.M)
    return {"commits": commits, "modified_tracked_files": dirty, "environment": env,
            "first_start": starts[0] if starts else "", "last_start": starts[-1] if starts else ""}  # fmt: skip


def provenance(out: Path, machine: str) -> dict:
    """
    Everything the record states about where its numbers came from. ``commit`` is the commit the
    runs started at when they all started at one, else the checkout at recording; the record is
    named after it.
    """
    runs = run_settings(out)
    head = runs["commits"][0] if len(runs["commits"]) == 1 else git("rev-parse", "HEAD")
    libraries = {}
    for name in LIBRARIES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            libraries[name] = metadata.version(name)
    try:
        import psutil

        ram_gib = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        ram_gib = None
    return {
        "recorded": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": head,
        "commit_date": git("show", "-s", "--format=%cI", head),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "checkout_at_recording": git("rev-parse", "HEAD"),
        "modified_tracked_files": len([line for line in git("status", "--porcelain", "--untracked-files=no").splitlines() if line]),
        "uv_lock_blob": git("show", f"{head}:uv.lock") and run(["git", "-C", str(ROOT), "rev-parse", f"{head}:uv.lock"]),
        "runs": runs,
        "machine": {"label": machine, "os": platform.platform(), "arch": platform.machine(), "cpu": cpu_model(),
                    "cores": os.cpu_count(), "ram_gib": ram_gib, "gpus": gpus(), "cuda_runtime": cuda_runtime()},
        "python": platform.python_version(),
        "libraries": libraries,
        "mpi_library": mpi_library(),
        "threads": {k: os.environ.get(k, "") for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }  # fmt: skip


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if value != value else value


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if value == int(value) and abs(value) >= 1:
        return f"{int(value):,}"
    return f"{value:.3g}"


def results_rows(comparison: list[dict]) -> list[dict]:
    """The collector's comparison rows as the record's columns: the value measured here first."""
    rows = []
    for r in comparison:
        rows.append({
            "table": r["ref"].split(" #")[0], "row": r["ref"].split(" #")[1] if " #" in r["ref"] else "",
            "location": r["loc"], "problem": r["problem"], "d": r["dim"], "tol": r["tol"], "quantity": r["quantity"],
            "value": r["rerun_seed0"], "n_seeds": r["n_seeds"], "seed_min": r["seed_min"], "seed_max": r["seed_max"],
            "paper_value": r["paper"], "relative_difference_to_paper": r["rel_diff"],
        })  # fmt: skip
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def markdown(prov: dict, rows: list[dict], checks: list[dict]) -> str:
    m, runs = prov["machine"], prov["runs"]
    gpu = "; ".join(f"{g['name']} ({g['memory']}, driver {g['driver']})" for g in m["gpus"]) or "none"
    lines = [
        f"# Manuscript results, {prov['recorded'][:10]}, commit `{prov['commit'][:7]}`",
        "",
        "The numbers the manuscript reports, as the code at this commit measures them. The paper's own",
        "values, the seed replicates and the relative differences are in `manuscript_results.csv`.",
        "",
        "| | |",
        "|---|---|",
        f"| Commit | `{prov['commit']}` (committed {prov['commit_date']}; branch at recording: {prov['branch']}) |",
        f"| Runs started at | {', '.join(f'`{c[:7]}`' for c in runs['commits']) or 'unknown'}"
        f" ({runs['first_start']} to {runs['last_start']}) |",
        f"| Modified tracked files | {', '.join(map(str, runs['modified_tracked_files'])) or 'unknown'} when the runs started |",
        f"| Runs on the GPU | {prov['gpu_runs']} |",
        f"| Machine | {m['label']}: {m['cpu']}, {m['cores']} cores, {m['ram_gib']} GiB RAM, {m['os']} |",
        f"| GPU | {gpu}; CUDA runtime {m['cuda_runtime'] or 'none'} |",
        f"| Python and libraries | Python {prov['python']}; "
        + ", ".join(f"{k} {v}" for k, v in prov["libraries"].items()) + " |",
        f"| MPI | {prov['mpi_library'] or 'none'} |",
        f"| `uv.lock` | blob `{prov['uv_lock_blob'][:12]}` |",
        f"| Run settings | {'; '.join(runs['environment']) or 'unknown'} |",
        "",
    ]  # fmt: skip
    if len(runs["commits"]) > 1:
        lines += ["**The runs started at more than one commit**; check that the code they ran did not change.", ""]
    if prov["checkout_at_recording"] != prov["commit"]:
        lines += [f"Recorded from a checkout of `{prov['checkout_at_recording'][:7]}`; the numbers are the runs' own.", ""]
    by_table: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in rows:
        key = (r["row"], r["problem"], r["d"], r["tol"])
        by_table[r["table"]].setdefault(key, {})[r["quantity"]] = number(r["value"])
    for table, entries in by_table.items():
        present = [q for q in QUANTITIES if any(q in v for v in entries.values())]
        lines += [f"## {table}", "", "| # | problem | d | tol | " + " | ".join(present) + " |",
                  "|---|---|---|---|" + "---|" * len(present)]  # fmt: skip
        def order(key):  # rows are numbered, some with a label after the number ("1 (AFINNI)")
            digits = re.match(r"\d+", key[0])
            return (int(digits.group()) if digits else 0, key[0], float(key[3] or 0))

        for (row, problem, d, tol), values in sorted(entries.items(), key=lambda kv: order(kv[0])):
            lines.append(f"| {row} | {problem} | {d} | {tol} | " + " | ".join(fmt(values.get(q)) for q in present) + " |")
        lines.append("")
    counts = defaultdict(int)
    for c in checks:
        counts[c["result"]] += 1
    lines += ["## Accuracy checks", "", ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none", ""]
    for c in checks:
        if c["result"] == "FAIL":
            lines.append(f"- FAIL {c['ref']}: {c['check']} (value {c['value']}, limit {c['limit']}) {c['note']}".rstrip())
    measured = sum(r["value"] not in ("", "nan") for r in rows)
    lines += ["", f"{measured} of {len(rows)} reported numbers measured.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the run_paper.sh --out folder, collected")
    parser.add_argument("--machine", required=True, help='a neutral description, e.g. "GH200 node"')
    parser.add_argument("--dest", type=Path, default=DEST)
    a = parser.parse_args()
    report = a.out / "report"
    if not (report / "comparison.csv").exists():
        sys.exit(f"{report / 'comparison.csv'} is missing: run collect_paper.py --out {a.out} first")
    prov = provenance(a.out, a.machine)
    devices = [r.get("gpu", "") for r in read_csv(report / "runs.csv")]
    prov["gpu_runs"] = f"{sum(v == 'True' for v in devices)} of {len(devices)}"
    slug = re.sub(r"[^a-z0-9]+", "-", a.machine.lower()).strip("-")
    record = a.dest / f"{prov['recorded'][:10]}_{prov['commit'][:7]}_{slug}"
    record.mkdir(parents=True, exist_ok=True)
    rows = results_rows(read_csv(report / "comparison.csv"))
    write_csv(record / "manuscript_results.csv", rows)
    for name in ("slopes.csv", "checks.csv", "cost.csv"):
        if (report / name).exists():
            shutil.copyfile(report / name, record / name)
    (record / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    checks = read_csv(report / "checks.csv")
    (record / "RESULTS.md").write_text(markdown(prov, rows, checks))
    index = a.dest / "INDEX.md"
    if not index.exists():
        index.write_text("# Recorded manuscript results\n\nOne row per recording (`record_results.py`).\n\n"
                         "| Recorded | Commit | Machine | Numbers measured | Record |\n|---|---|---|---|---|\n")  # fmt: skip
    measured = sum(r["value"] not in ("", "nan") for r in rows)
    with index.open("a") as fh:
        fh.write(f"| {prov['recorded'][:10]} | `{prov['commit'][:7]}` | {a.machine} | {measured} of {len(rows)} | "
                 f"[{record.name}]({record.name}/RESULTS.md) |\n")  # fmt: skip
    print(f"recorded {measured} of {len(rows)} numbers in {record}")
    if any(prov["runs"]["modified_tracked_files"]):
        print("warning: the tree had modified tracked files when the runs started; the commit does not fully describe the code")
    if len(prov["runs"]["commits"]) > 1:
        print(f"warning: the runs started at {len(prov['runs']['commits'])} different commits")


if __name__ == "__main__":
    main()
