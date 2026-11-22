"""
Learning curves in whole trajectories (council round 3: Fable's first action). Is an unserved problem limited by its data? The fourth recipe
(generic4.py, R4e's chart dimension) trained on the benchmark's 100 training trajectories plus 120, 240 or 480 more (totals 220, 340, 580),
drawn from the problem's own initial law with new seeds (11, 12, ...; the benchmark uses 1 and 2), burnt in and driven like the
benchmark's (src.benchmark.step_maps.data), scored on the same 40 test trajectories. The extra trajectories are records under
data/records/p13_data/ (one per problem, the solver seconds they cost kept with them); the 100-trajectory point is R4e's record. One record
per (problem, total) under data/records/p13/; a case is limited to 1,800 s. Usage: p13_learning.py [problem ...].

Predictions, committed before the run:
  L1  POLLU is served at 340 trajectories, its 40 test rollouts within the tolerance (Fable's check: a 39-to-2.8e-3 drop of the ungated
      rollout with 240 more trajectories);
  L2  Nelson's best held-out calibration worst falls at least 3x from 100 to 580 trajectories, and Nelson is served at 580;
  L3  primordial is not served at any size (its one-step jumps of up to 11 dex are not a data limit);
  L4  HIRES is not served at any size (the product gate refuses it, whatever the data);
  L5  allen_cahn_d5 is served at 340 or more;
  L6  lorenz96_d20, the Oregonator and Van der Pol are not served at 580.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from generic4 import run

import src.problems.ode as ode
from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
PROBLEMS = ("pollu", "allen_cahn_d5", "nelson", "primordial", "hires", "lorenz96_d20", "oregonator", "van_der_pol")
EXTRA = (120, 240, 480)
CHUNK = 120


def chunk(name: str, seed: int) -> dict:
    """CHUNK trajectories of ``name`` from seed ``seed``, as the benchmark draws its own."""
    p = STEP_MAPS[name]
    t0 = time.perf_counter()
    Y = p.initial(CHUNK, seed)
    if p.burn > 0:
        Y = p.system.step(Y, p.burn)
    path = p.held_path(CHUNK, seed, p.steps) if p.held_path else None
    Z = ode.trajectories(p.system, Y, p.dt, p.steps, path)
    ok = np.all(np.isfinite(Z), axis=(1, 2))
    return {"solver_s": time.perf_counter() - t0, "kept": int(ok.sum()), "_arrays": {"Z": Z[ok]}}


def extra_trajectories(name: str, n: int, store: RecordStore) -> tuple[np.ndarray, float]:
    Zs, secs = [], 0.0
    for c in range(-(-n // CHUNK)):
        rec = recorded(store, {"problem": name, "chunk": c, "seed": 11 + c, "size": CHUNK}, chunk, name, 11 + c, limit=3600)
        if rec.get("status") != "done":
            raise RuntimeError(f"extra trajectories {name} {c}: {rec.get('failure', {}).get('message')}")
        Zs.append(np.load(store.path(rec["key"]).with_suffix(".npz"))["Z"])
        secs += rec["solver_s"]
    return np.concatenate(Zs)[:n], secs


def case(name: str, n_extra: int, dstore: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    extra, secs = extra_trajectories(name, n_extra, dstore)
    ref = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    out = run(P, np.concatenate([train, extra]), test, name, solve_us=baseline_cost(name)["solve_us"], k=(ref["k"], ref["decoder_degree"]))
    out["trajectories"] = int(len(train) + len(extra))
    out["extra_solver_s"] = secs
    out["best_heldout_worst"] = min(r["heldout_worst"] for r in out["selection"])
    log(f"  {name} {out['trajectories']} trajectories: served {out['served']}, q {out['degree']}, best held-out worst {out['best_heldout_worst']:.2e}, "
        f"rollout {out['rollout_worst']:.2e}, refused {out['refused_percent']:.1f}% (extra data {secs:.0f} solver s)")  # fmt: skip
    return out


def main() -> None:
    names = sys.argv[1:] or list(PROBLEMS)
    dstore = RecordStore(OUT / "records" / "p13_data", path_of=lambda k: OUT / "records" / "p13_data" / f"{k['problem']}_{k['chunk']}.json")
    store = RecordStore(OUT / "records" / "p13", path_of=lambda k: OUT / "records" / "p13" / f"{k['problem']}_{k['trajectories']}.json")
    fields = ("status", "trajectories", "extra_solver_s", "served", "degree", "best_heldout_worst", "rollout_worst", "refused_percent",
              "mixing_chained", "speedup_effective", "train_s")  # fmt: skip
    for name in names:
        for n_extra in EXTRA:
            recorded(store, {"problem": name, "trajectories": 100 + n_extra, "tol": 0.01}, case, name, n_extra, dstore, limit=1800)
    summary = {f"{r['key']['problem']}/{r['key']['trajectories']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "p13_learning.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
