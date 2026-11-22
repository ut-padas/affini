"""
Rollout-tuned ChebTT under chained mixing (council: Fable's C1). The benchmark's `chebtt_roll` (ChebTT retrained until its rollouts hold)
is rebuilt, exactly as the benchmark builds it (src.benchmark.step_maps.METHODS, tolerance 0.01, seed 1), on every step map where its
benchmark record holds the tolerance in rollouts and trained within 300 s; nothing is written to the benchmark's store. Scored as the
generic recipe is: the worst rollout over the benchmark's 20 test trajectories, and the chained mixing rollouts (w = 0.01, 0.1; refused
inputs stepped by the solver) with their solver shares. One record per problem under data/records/p4/; a case is limited to 900 s.

Prediction, committed before the run (Fable's, council-fable.md C1): a convex mixture of in-box states stays in ChebTT's box, so mixing is
not off its data: chebtt_roll holds the tolerance under chained mixing at both weights on at least 13 of the problems it serves, with a
solver share under mixing within two percentage points of its unmixed one.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from generic import mixing

from src import PROJECT_DIR
from src.benchmark.step_maps import METHODS, STEP_MAPS, data, rollout
from src.benchmark.table import RESULTS_CHANGED
from src.infra.records import RecordStore, stale, upgrade

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01


def candidates() -> list[str]:
    out = []
    for f in glob.glob(str(PROJECT_DIR / "results" / "benchmark" / "runs" / "step_maps" / "cpu-1rank" / "*" / "*" / "chebtt_roll" / "tol0.01-s1.json")):
        r = upgrade(json.loads(Path(f).read_text()))
        if r.get("status") == "done" and "failed" not in r and not stale(r, RESULTS_CHANGED) and (r.get("rollout_worst") or 1) <= TOL and (r.get("train_s") or 1e9) <= 300:
            out.append(r["problem"])
    return sorted(out, key=lambda n: (data(n)[0].shape[2], n))


def case(name: str) -> dict:
    p = STEP_MAPS[name]
    train, test = data(name)
    model = METHODS["chebtt_roll"][2](p, train, TOL, 1)
    r20 = rollout(p, model, test[:20])
    mix = {str(w): mixing(p, model, test, w, np.random.default_rng(1)) for w in (0.01, 0.1)}
    out = {"evals": int(getattr(model, "evals", 0)), "rollout_worst_20": float(r20["rollout_worst"]),
           "rollout_solver_share_20": float(np.mean(r20["rollout_solved_by_step"])), "mixing_chained": mix}  # fmt: skip
    log(f"  {name}: rollout {out['rollout_worst_20']:.2e} (solver {100 * out['rollout_solver_share_20']:.1f}%); chained mixing 0.01 "
        f"{mix['0.01']['worst']:.2e} ({100 * mix['0.01']['solver_share']:.0f}%), 0.1 {mix['0.1']['worst']:.2e} ({100 * mix['0.1']['solver_share']:.0f}%)")  # fmt: skip
    return out


def main() -> None:
    names = sys.argv[1:] or candidates()
    log(f"chebtt_roll candidates: {names}")
    store = RecordStore(OUT / "records" / "p4", path_of=lambda key: OUT / "records" / "p4" / f"{key['problem']}.json")
    for name in names:
        recorded(store, {"problem": name, "method": "chebtt_roll", "tol": TOL}, case, name, limit=900)
    summary = {r["key"]["problem"]: {k: r.get(k) for k in ("status", "evals", "rollout_worst_20", "rollout_solver_share_20", "mixing_chained")} for r in store.records()}
    (OUT / "p4_chebtt_mixing.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
