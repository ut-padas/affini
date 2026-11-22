"""
The existing schemes' benchmark results on the step maps at tolerance 0.01, beside the generic recipe's (R4b), in one table for review:
per problem and method, the status (with the failure's reason), the worst one-step error over accepted test inputs, the worst rollout over
the benchmark's 20 test trajectories, the refused share and the rollout's solver share, the solver calls to build, the training time, and
the nominal and effective speed-ups (Corollary 5.3 (c): solve / ((1 - rho) predict + rho solve), the gate's cost inside the recorded
prediction time). Records are read only from the benchmark store (results/benchmark/runs/, seed 1); a record older than its method's
last change to results (src/benchmark/table.py's RESULTS_CHANGED) is marked stale. The recipe's rows use its 20-trajectory rollout (the
benchmark's cohort), and its solves count only its lift checks: like SlowTable it learns from the training trajectories every method is
given, which no row counts. Writes data/scheme_table.json and data/scheme_table.md.
"""

import glob
import json
from pathlib import Path

import numpy as np

from src import PROJECT_DIR
from src.benchmark.table import RESULTS_CHANGED
from src.infra.records import stale, upgrade

OUT = Path(__file__).resolve().parent / "data"


def row(r: dict) -> dict:
    out = {"status": r.get("status", "done") if "failed" not in r else "failed", "stale": bool(stale(r, RESULTS_CHANGED))}
    if out["status"] != "done":
        out["reason"] = str(r.get("failed", ""))[:80]
        return out
    rho = (r.get("refused_percent") or 0.0) / 100
    solved = r.get("rollout_solved_by_step") or []
    out.update(worst=r.get("worst"), rollout_worst=r.get("rollout_worst"), refused_percent=r.get("refused_percent"),
               rollout_solver_share=float(np.mean(solved)) if solved else None, evals=r.get("evals"), train_s=r.get("train_s"),
               speedup_nominal=r.get("speedup"))  # fmt: skip
    if r.get("solve_us") and r.get("predict_us"):
        out["speedup_effective"] = r["solve_us"] / ((1 - rho) * r["predict_us"] + rho * r["solve_us"])
    return out


def g(x, f="{:.1e}"):
    return "–" if x is None else f.format(x)


def main() -> None:
    table: dict = {}
    for f in glob.glob(str(PROJECT_DIR / "results" / "benchmark" / "runs" / "step_maps" / "cpu-1rank" / "*" / "*" / "*" / "tol0.01-s1.json")):
        r = upgrade(json.loads(Path(f).read_text()))
        table.setdefault(r["problem"], {})[r["method"]] = row(r)
    for f in glob.glob(str(OUT / "records" / "r4b" / "*.json")):
        r = json.loads(Path(f).read_text())
        if r.get("status") != "done":
            continue
        table.setdefault(r["key"]["problem"], {})["generic2 (R4b)"] = {
            "status": "done" if r["served"] else "not served", "worst": r["one_step_worst"], "rollout_worst": r["rollout_worst_20"],
            "rollout_worst_40": r["rollout_worst"], "refused_percent": r["refused_percent"], "rollout_solver_share": r["rollout_solver_share_20"],
            "evals": r.get("k_solves"),
            "train_s": r.get("train_s"), "speedup_nominal": r.get("speedup_nominal"), "speedup_effective": r.get("speedup_effective"),
            "mixing_chained_0.01": r["mixing_chained"]["0.01"]["worst"], "mixing_chained_0.1": r["mixing_chained"]["0.1"]["worst"]}  # fmt: skip
    (OUT / "scheme_table.json").write_text(json.dumps(table, indent=1, sort_keys=True, default=str) + "\n")
    lines = ["| problem | method | status | one-step worst | rollout worst (20) | refused % | rollout solver share | solves to build | train s | speed-up nominal / effective |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for p in sorted(table):
        for m in sorted(table[p]):
            x = table[p][m]
            st = x["status"] + (" (stale)" if x.get("stale") else "") + (f": {x['reason']}" if x.get("reason") else "")
            lines.append(f"| {p} | {m} | {st} | {g(x.get('worst'))} | {g(x.get('rollout_worst'))} | {g(x.get('refused_percent'), '{:.1f}')} | "
                         f"{g(x.get('rollout_solver_share'), '{:.2f}')} | {g(x.get('evals'), '{:,}')} | {g(x.get('train_s'), '{:.0f}')} | "
                         f"{g(x.get('speedup_nominal'), '{:.0f}')} / {g(x.get('speedup_effective'), '{:.0f}')} |")  # fmt: skip
    (OUT / "scheme_table.md").write_text("\n".join(lines) + "\n")
    print(f"{len(table)} problems, {sum(len(v) for v in table.values())} rows")


if __name__ == "__main__":
    main()
