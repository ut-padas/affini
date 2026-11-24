"""
The paper's tables, declared as data and built from collect.py's rows.

A table selects the finished cases of one experiment (``where``), groups them by its row keys
(``by``) and gives each column a source: a column of cases.csv, aggregated over the seeds of a
group, optionally from other cases that share the row keys (``also``: e_nh from the runs without
Hessian splits, the exact-gradient columns, the static timings). A table can therefore gain, drop
or rename a column by editing TABLES, without rerunning anything. Every column is written as its
median over seeds with its minimum, maximum and the seed count beside it.

The convergence, dimension and absolute-tolerance tables have one row per method (AFINNI, ART with
affine leaves, ART with quadratic leaves) for each setting. Their p_o and e_in columns take AFINNI's
rows only (``AFINNI``): an ART model routes every query in the domain to a leaf, so it has no
outside share and no inside error apart from e, and its cells stay empty ("–"). The other tables
are AFINNI's alone.

AFINNI's predictions are the candidate search's (``cand.``: the nearest leaf among the CANDIDATES
nearest centres), as in the paper. The same quantities completed by the exact search (``exact.``)
are the columns ending in ``_exact_search``, for AFINNI's rows only: an ART model's routed
prediction is the same under both. train_s is the training's wall time and infer_us the inference
time per test point in microseconds.
"""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

AFINNI = {"method": "afinni"}
# AFINNI's predictions completed by the exact search, beside the candidate search's columns
EXACT_SEARCH = [("p_o_exact_search", "exact.p_o", AFINNI), ("e_in_exact_search", "exact.e_in", AFINNI),
                ("e_exact_search", "exact.e", AFINNI), ("infer_us_exact_search", "exact.us_per_query", AFINNI)]
# name -> where (case filters), by (row keys), columns (header, cases.csv column, extra filters)
TABLES = {
    "convergence": {
        "where": {"experiment": "convergence", "mode": "hessian"}, "by": ("function", "d", "tol", "method"),
        "columns": [("N", "N", {}), ("N_c", "N_c", {}), ("p_o", "cand.p_o", AFINNI), ("e_in", "cand.e_in", AFINNI),
                    ("e", "cand.e", {}), ("e_nh", "cand.e", {"mode": "no_hessian"}), ("train_s", "train_s", {}),
                    ("infer_us", "cand.us_per_query", {}), *EXACT_SEARCH],
    },
    "dimension": {
        "where": {"experiment": "dimension"}, "by": ("d", "method"),
        "columns": [("N", "N", {}), ("N_c", "N_c", {}), ("p_o", "cand.p_o", AFINNI), ("e_in", "cand.e_in", AFINNI),
                    ("e", "cand.e", {}), ("train_s", "train_s", {}), ("infer_us", "cand.us_per_query", {}),
                    *EXACT_SEARCH],
    },
    "absolute": {
        "where": {"experiment": "absolute"}, "by": ("atol", "n0", "method"),
        "columns": [("N", "N", {}), ("N_c", "N_c", {}), ("p_o", "cand.p_o", AFINNI), ("e_in", "cand.e_in", AFINNI),
                    ("e", "cand.e", {}), ("p_in_good", "cand.p_in_good", {}), ("train_s", "train_s", {}),
                    ("infer_us", "cand.us_per_query", {}), *EXACT_SEARCH,
                    ("p_in_good_exact_search", "exact.p_in_good", AFINNI)],
    },
    "gradient": {
        "where": {"experiment": "convergence", "function": "wave", "d": 5, "mode": "hessian", **AFINNI}, "by": ("tol",),
        "columns": [("N_c", "N_c", {}), ("p_o", "cand.p_o", {}), ("e_in", "cand.e_in", {}), ("e", "cand.e", {}),
                    ("N_c_exact_gradient", "N_c", {"mode": "exact_gradient", "experiment": "gradient"}),
                    ("p_o_exact_gradient", "cand.p_o", {"mode": "exact_gradient", "experiment": "gradient"}),
                    ("e_in_exact_gradient", "cand.e_in", {"mode": "exact_gradient", "experiment": "gradient"}),
                    ("e_exact_gradient", "cand.e", {"mode": "exact_gradient", "experiment": "gradient"})],
    },
    "distributed": {
        "where": {"experiment": "distributed", "work_stealing": True, **AFINNI}, "by": ("tol", "ranks"),
        "columns": [("dynamic_s", "train_s", {}), ("static_s", "train_s", {"work_stealing": False}),
                    ("N_c", "N_c", {}), ("e_in", "cand.e_in", {})],
    },
}
PROGRESSION = {"function": "rosenbrock", "d": 10, "tol": 0.05, "mode": "hessian", "seed": 0, **AFINNI}


def number(v):
    """A cases.csv value as a number, or None (empty, text, NaN)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def matches(row: dict, where: dict) -> bool:
    for k, want in where.items():
        if k == "experiment":
            if want not in str(row.get("experiments", "")).split():
                return False
        elif isinstance(want, bool):
            if str(row.get(k)).lower() not in (str(want).lower(), "1" if want else "0"):
                return False
        elif isinstance(want, (int, float)):
            if number(row.get(k)) != float(want):
                return False
        elif str(row.get(k)) != str(want):
            return False
    return True


def aggregate(values: list[float]) -> dict:
    if not values:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


def build_table(cases: list[dict], spec: dict) -> list[dict]:
    """One row per group of ``spec['by']``: every column's median, min, max and seed count over finished cases."""
    done = [r for r in cases if r.get("status") == "done"]
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for header, source, extra in spec["columns"]:
        where = {**spec["where"], **extra}
        for r in done:
            if matches(r, where):
                key = tuple(r.get(k) for k in spec["by"])
                v = number(r.get(source))
                if v is not None:
                    groups[key][header].append(v)
    out = []
    for key in sorted(groups, key=lambda k: tuple(number(x) if number(x) is not None else str(x) for x in k)):
        row = dict(zip(spec["by"], key, strict=True))
        for header, _source, _extra in spec["columns"]:
            agg = aggregate(groups[key].get(header, []))
            row[header], row[f"{header}.min"], row[f"{header}.max"], row[f"{header}.seeds"] = agg.values()
        out.append(row)
    if "static_s" in [h for h, *_ in spec["columns"]]:  # a derived column: the speedup of work stealing
        for row in out:
            row["speedup"] = row["static_s"] / row["dynamic_s"] if row["static_s"] and row["dynamic_s"] else None
    return out


def build(cases: list[dict], levels: list[dict], rounds: list[dict], folder: Path) -> list[str]:
    """Write every table to ``folder`` as <name>.csv and all of them to tables.json; returns their names."""
    folder.mkdir(parents=True, exist_ok=True)
    built = {name: build_table(cases, spec) for name, spec in TABLES.items()}
    key = next((r["key"] for r in cases if r.get("status") == "done" and all(
        (number(r.get(k)) == v if isinstance(v, (int, float)) else str(r.get(k)) == v) for k, v in PROGRESSION.items())), None)  # fmt: skip
    built["progression"] = [
        {"level": lv["level"], "clusters": lv.get("clusters_start"), "points": lv.get("points_start"),
         "time_s": lv.get("runtime_sec"), "host_gib": lv.get("host_rss_gib"), "device_gib": lv.get("device_pool_gib")}
        for lv in levels if lv["key"] == key]  # fmt: skip
    by_round: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rd in rounds:
        for q in ("N_train", "N_c", "p_o", "e_in", "e"):
            v = number(rd.get(q))
            if v is not None:
                by_round[(rd["function"], int(rd["round"]))][q].append(v)
    built["iterative"] = [{"function": f, "round": i, **{q: aggregate(v)["median"] for q, v in qs.items()},
                           "seeds": max(len(v) for v in qs.values())} for (f, i), qs in sorted(by_round.items())]  # fmt: skip
    for name, data in built.items():
        with (folder / f"{name}.csv").open("w", newline="") as fh:
            fields = list(dict.fromkeys(k for row in data for k in row)) or ["empty"]
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    (folder / "tables.json").write_text(json.dumps(built, indent=1) + "\n")
    return list(built)
