"""
Collect the manuscript study's results into data files and build its tables.

    python -m studies.manuscript.collect [--results studies/manuscript/output] [--data <folder>]

Reads ``<results>/cases/*.json`` (case.py, run.py) and writes to ``<data>`` (default
``<results>/data``):

- ``cases.csv``: one row per case, with its settings (``method``: afinni, art_affine, art_quadratic),
  its status (done, failed, skipped) and the reason, and every scalar it recorded; the candidate
  search's and the exact search's metrics are prefixed ``cand.`` and ``exact.`` (with each one's
  inference time per test point, ``us_per_query``, in microseconds), and an ART case's
  leaves by terminal reason and its test points in unaccepted leaves ``art.``;
- ``levels.csv``: one row per case and level (rank 0's levels for multi-rank cases);
- ``rounds.csv``: one row per iterative-refinement case and round;
- ``provenance.json``: the commits, machines and library versions the results came from;
- ``tables/``: the paper's tables (tables.py), as CSV and as one JSON for the paper's Typst source.

Everything here reads files only, so it runs on a login node and again whenever the tables change.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from studies.manuscript import tables  # noqa: E402

CASE_FIELDS = ("key", "method", "function", "d", "mode", "seed", "n0", "k0", "m", "tol", "atol", "ranks", "work_stealing", "rounds")
RUN_FIELDS = ("N", "N_c", "num_levels", "generated_points", "converged", "unresolved_points", "train_s", "predict_s",
              "exact_s", "wall_s")  # fmt: skip


def metric_columns(prefix: str, m: dict | None) -> dict:
    """The scalar metrics of one prediction (compute_error_metrics plus p_o), flattened under ``prefix``."""
    if not m:
        return {}
    n_in = m["n_test"] - m["n_out"]
    out = {f"{prefix}.p_o": m["p_o"], f"{prefix}.p_in_good": 100.0 * m["n_in_good"] / n_in if n_in else None,
           f"{prefix}.n_test": m["n_test"], f"{prefix}.n_out": m["n_out"]}  # fmt: skip
    for part, short in (("in", "in"), ("all", "all"), ("out", "out")):
        for k, v in m[part].items():
            out[f"{prefix}.{short}.{k}"] = v
    out[f"{prefix}.e_in"] = m["in"]["l2_rel_error"]
    out[f"{prefix}.e"] = m["all"]["l2_rel_error"]
    ratio = m.get("tolerance_ratio") or {}
    out[f"{prefix}.max_ratio_in"], out[f"{prefix}.max_ratio_out"] = ratio.get("max_in"), ratio.get("max_out")
    return out


def rows(results: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """(cases, levels, rounds, provenances) from every result file."""
    cases, levels, rounds, provs = [], [], [], []
    for path in sorted((results / "cases").glob("*.json")):
        r = json.loads(path.read_text())
        c = r["case"]
        # a result written before the study ran ART has no method: it is AFINNI's
        row = {k: c.get(k) for k in CASE_FIELDS} | {"method": c.get("method", "afinni")} | {"experiments": " ".join(c.get("experiments", ())),
                                                    "status": r["status"]}  # fmt: skip
        fail = r.get("failure") or {}
        row |= {"stage": r.get("stage"), "failure": fail.get("class"), "reason": r.get("reason") or fail.get("message")}
        row |= {k: r.get(k) for k in RUN_FIELDS}
        mem = r.get("memory") or {}
        row |= {"host_peak_gib": mem.get("host_peak_gib"), "device_peak_gib": mem.get("device_peak_gib")}
        row |= metric_columns("cand", r.get("candidates")) | metric_columns("exact", r.get("exact"))
        for prefix, seconds in (("cand", r.get("predict_s")), ("exact", r.get("exact_s"))):
            if seconds is not None and row.get(f"{prefix}.n_test"):  # inference cost per query, comparable across test sets
                row[f"{prefix}.us_per_query"] = 1e6 * seconds / row[f"{prefix}.n_test"]
        art = r.get("art") or {}
        row |= {f"art.{k}": v for k, v in (art.get("terminal_reasons") or {}).items()}
        row |= {"art.p_unaccepted": art.get("p_unaccepted")} if art else {}
        prov = r.get("provenance") or {}
        row |= {"commit": (prov.get("commit") or "")[:12], "gpu": (prov.get("gpu") or {}).get("name"), "cpu": prov.get("cpu")}
        cases.append(row)
        for lv in r.get("level_stats") or []:
            if lv.get("rank", 0) == 0:
                levels.append({"key": c["key"], **lv})
        for i, rd in enumerate(r.get("rounds") or []):
            rounds.append({"key": c["key"], "function": c["function"], "seed": c["seed"], "round": i, **rd})
        if prov:
            provs.append(prov)
    return cases, levels, rounds, provs


def write_csv(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "studies" / "manuscript" / "output")
    parser.add_argument("--data", type=Path, default=None)
    a = parser.parse_args(argv)
    data = a.data or a.results / "data"
    cases, levels, rounds, provs = rows(a.results)
    write_csv(data / "cases.csv", cases)
    write_csv(data / "levels.csv", levels)
    write_csv(data / "rounds.csv", rounds)
    seen = {}
    for p in provs:  # one entry per (commit, machine): what produced the data
        k = (p.get("commit"), p.get("cpu"), (p.get("gpu") or {}).get("name"))
        seen.setdefault(k, {key: p.get(key) for key in ("commit", "tracked_changes", "python", "libraries", "machine",
                                                         "cpu", "cores", "ram_gib", "gpu")} | {"cases": 0})  # fmt: skip
        seen[k]["cases"] += 1
    (data / "provenance.json").write_text(json.dumps(list(seen.values()), indent=1) + "\n")
    built = tables.build(cases, levels, rounds, data / "tables")
    states = {}
    for row in cases:
        states[row["status"]] = states.get(row["status"], 0) + 1
    print(f"{len(cases)} cases ({', '.join(f'{k} {v}' for k, v in sorted(states.items()))}); tables: {', '.join(built)}; in {data}")


if __name__ == "__main__":
    main()
