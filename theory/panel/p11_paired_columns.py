"""
Paired columns (council round 2: Astra's controlled comparison, Fable's D = 0.25 and unsorted cells): does a surrogate's error feed back
through the coupling? p6's column (the test trajectories as cells, conservative transport between neighbours, then chemistry) in two
neighbour modes with everything else identical (geometry, drivers, initial states, exchange):
  closed  a cell's neighbours are the hybrid column's own cells (their surrogate errors enter it through transport);
  forced  a cell's neighbours are the reference column's cells (only its own error propagates; the reference-partner control without
          p6's change of donor geometry);
each with the cells ordered along their first principal coordinate (sorted, as p6) or in a random order (unsorted: neighbours unlike
each other, so transport moves cells off the bundle), at D = 0.005, 0.05 and 0.25. Per step, the worst error splits into the cell's own
one-step defect on its accepted input (surrogate against solver from the same input) and the inherited error (the solver from the hybrid
input against the reference column). Surrogates: the fourth recipe on every problem R4e serves (rebuilt from its record). One record per
problem under data/records/p11/; a case is limited to 900 s. Usage: p11_paired_columns.py [problem ...].

Predictions, committed before the run:
  C1  closed and forced columns agree within a factor 1.5 in their worst error at every exchange and order wherever both hold 0.01
      (transport averages errors of both signs, so feedback through neighbours does not compound at these exchanges);
  C2  the unsorted column at D = 0.25 has the largest own-input defect of the six settings on at least 7 of the 10 problems (its inputs
      lie farthest off the bundle);
  C3  every R4e-served surrogate holds 0.01 in all twelve settings except possibly at unsorted D = 0.25.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from generic4 import rebuild, verified_laws

from src.benchmark.step_maps import STEP_MAPS, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
TOL = 0.01


def worst(P, Za, Zb):
    e = np.where(P.scored(Zb), np.abs(Za - Zb), 0.0)
    return float(np.max(e)) if np.all(np.isfinite(e)) else float("inf")


def paired(P, sur, test, D, mode, order, W) -> dict:
    h = P.system.held
    free = test.shape[2] - h
    X0 = test[:, 0, :free] - test[:, 0, :free].mean(0)
    perm = np.argsort(X0 @ np.linalg.svd(X0, full_matrices=False)[2][0]) if order == "sorted" else np.random.default_rng(4).permutation(len(test))
    true = test[perm]
    lin = (lambda Z: 10.0**Z) if P.system.log else (lambda Z: Z)
    back = (lambda Y: np.log10(np.maximum(Y, 1e-300))) if P.system.log else (lambda Y: Y)

    def transport(Z, donors, k):
        X, Xd = lin(Z[:, :free]), lin(donors[:, :free])
        Xp = np.concatenate([Xd[:1], Xd, Xd[-1:]])
        out = Z.copy()
        out[:, :free] = back((1 - 2 * D) * X + D * Xp[:-2] + D * Xp[2:])
        if h:
            out[:, free:] = true[:, k, free:]
        return out

    Zt, Zs = true[:, 0].copy(), true[:, 0].copy()
    by_step, own_by_step, inherited_by_step, solved = [], [], [], []
    for k in range(true.shape[1] - 1):
        Zin = transport(Zs, Zs if mode == "closed" else Zt, k)
        Zt = P.step(transport(Zt, Zt, k))
        r = np.asarray(sur.refuses(Zin), dtype=bool)
        nxt = Zin + sur.predict(Zin)
        exact = P.step(Zin)
        own_by_step.append(worst(P, nxt[~r], exact[~r]) if (~r).any() else 0.0)
        inherited_by_step.append(worst(P, exact, Zt))
        nxt[r] = exact[r]
        Zs = nxt
        by_step.append(worst(P, Zs, Zt))
        solved.append(float(r.mean()))
    out = {"worst": max(by_step), "own_defect_worst": max(own_by_step), "inherited_worst": max(inherited_by_step), "solver_share": float(np.mean(solved)),
           "worst_by_step": by_step, "own_by_step": own_by_step, "inherited_by_step": inherited_by_step}  # fmt: skip
    if len(W):
        mt, ms = lin(Zt[:, :free]) @ W[:, :free].T, lin(Zs[:, :free]) @ W[:, :free].T
        out["law_drift_over_scale"] = float(np.max(np.abs(ms - mt)) / max(np.max(np.abs(mt)), 1e-300))
    return out


def case(name: str) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    rec = json.loads((OUT / "records" / "r4e" / f"{name}.json").read_text())
    sur = rebuild(P, train, rec)
    W = verified_laws(P, train)
    out = {"settings": {}}
    t0 = time.perf_counter()
    for order in ("sorted", "unsorted"):
        for mode in ("closed", "forced"):
            for D in (0.005, 0.05, 0.25):
                c = paired(P, sur, test, D, mode, order, W)
                out["settings"][f"{order}/{mode}/{D}"] = c
                log(f"  {name} {order} {mode} D={D}: worst {c['worst']:.2e} (own {c['own_defect_worst']:.1e}, inherited {c['inherited_worst']:.1e}), "
                    f"solver {100 * c['solver_share']:.1f}%")  # fmt: skip
    out["wall_s"] = time.perf_counter() - t0
    return out


def main() -> None:
    import glob

    served = sorted(json.loads(Path(f).read_text())["key"]["problem"] for f in glob.glob(str(OUT / "records" / "r4e" / "*.json")) if json.loads(Path(f).read_text()).get("served"))
    names = sys.argv[1:] or served
    store = RecordStore(OUT / "records" / "p11", path_of=lambda key: OUT / "records" / "p11" / f"{key['problem']}.json")
    for name in names:
        recorded(store, {"problem": name, "method": "generic4", "tol": TOL}, case, name, limit=900)
    summary = {r["key"]["problem"]: {s: {k: v for k, v in c.items() if not k.endswith("_by_step")} for s, c in (r.get("settings") or {}).items()} | {"status": r.get("status")}
               for r in store.records()}  # fmt: skip
    (OUT / "p11_paired_columns.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
