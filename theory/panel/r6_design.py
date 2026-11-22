"""
Data design under the third recipe (council: Sol's controlled comparison, Fable's chemistry rungs; generic3.py, tolerance 0.01). The same
recipe with M fresh one-step pairs added to the training trajectories, drawn from a pool of 1,600 (r5_data_scaling.extra_pairs: starts
from each problem's initial law after its burn-in, stepped once by the solver), in five arms:
  m0          no extra pairs (R4c's record where it exists, with only the first-step scores computed here from its rebuilt surrogate;
              otherwise run here);
  random400   400 drawn uniformly from the pool, three draws (seeds 0-2);
  leverage400 400 chosen by sequential maximum leverage in the m0 configuration's fit space (the chart and degree m0 selected, fitted on
              the training trajectories): each pick is the pool state with the largest phi^T (A^T A)^-1 phi given the design so far,
              the square of Proposition 5.1's bound up to the factor N; it needs no solves, since the pool's labels are not looked at;
  all1600     the whole pool.
Everything but the extra pairs is fixed: the chart, the degree search, the calibration protocol (thirds, maximum, mixed), the margin.
Scores as generic3.run, and the step that every fit's failures sit at: the ungated worst and median one-step error on the test
trajectories' first step, and the share of first steps refused (the rebuilt served surrogate). One record per (problem, arm) under
data/records/r6/; a case is limited to 1,800 s. Usage: r6_design.py [problem ...].

Problems: Sol's four (fisher_kpp_d3, fisher_kpp_d10, allen_cahn_d10, burgers_d10), and the stiff chemistry R4c leaves for data
(chemakzo, hires, primordial, nelson, pollu; the three random draws only on the lattices, one on chemistry).

Predictions, committed before the run:
  E1  on the lattices, leverage400's ungated first-step worst is below the median of the three random400 draws on at least 3 of the 4
      problems, and at least 2x below it on at most 2 (Sol's threshold for adopting designed sampling is 2x);
  E2  all1600 serves every lattice problem with its 40 test rollouts within the tolerance, and the first-step refused share falls with M
      (m0 > random400 median > all1600) wherever m0 refuses any first step;
  E3  on chemistry, nelson and pollu are not served at any M (a total-degree space in 12-20 chart coordinates cannot reach the needed
      degree with at most 6,400 pairs: a basis-size limit, not a data limit), primordial is served at all1600 only if its R4c k is at most
      6, and hires and chemakzo are served at every M where R4c serves them;
  E4  no served arm has a test rollout above the tolerance (the margin and the calibrated gate hold under added data).
"""

import json
import sys
from pathlib import Path

import numpy as np
from generic import design, multi_indices
from generic2 import build
from generic3 import rebuild, run
from r5_data_scaling import extra_pairs

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import log, recorded  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
LATTICES = ("fisher_kpp_d3", "fisher_kpp_d10", "allen_cahn_d10", "burgers_d10")
CHEMISTRY = ("chemakzo", "hires", "primordial", "nelson", "pollu")
POOL = 1600


def arms(name: str) -> list[str]:
    draws = 3 if name in LATTICES else 1
    return ["m0", *[f"random400_s{s}" for s in range(draws)], "leverage400", "all1600"]


def m0_record(name: str, store: RecordStore) -> dict | None:
    """The configuration without extra data: R4c's record where it finished, otherwise this study's own m0."""
    r4c = OUT / "records" / "r4c" / f"{name}.json"
    if r4c.exists():
        rec = json.loads(r4c.read_text())
        if rec.get("status") == "done":
            return rec
    rec = None if store is None else store.read({"problem": name, "arm": "m0", "tol": 0.01})
    return rec if rec is not None and rec.get("status") == "done" else None


def leverage(P, train, rec: dict, Xpool: np.ndarray, m: int) -> np.ndarray:
    """Indices of ``m`` pool states by sequential maximum leverage in ``rec``'s fit space (module docstring), Sherman-Morrison updates."""
    sur, X, _ = build(P, train, rec["k"], rec["degree"], rec["ridge"], rec["decoder_degree"])
    U, Up = sur.T.chart(X, rec["k"]), sur.T.chart(Xpool, rec["k"])
    idx = multi_indices(U.shape[1], rec["degree"])
    lo, hi = U.min(0), U.max(0)
    A, Phi = design(U, rec["degree"], lo, hi, idx), design(Up, rec["degree"], lo, hi, idx)
    G = A.T @ A
    Ginv = np.linalg.pinv(G + 1e-12 * np.trace(G) / len(G) * np.eye(len(G)))  # a rank-deficient design still ranks the pool
    lev = np.einsum("ij,jk,ik->i", Phi, Ginv, Phi)
    chosen = []
    for _ in range(m):
        lev[chosen] = -np.inf
        j = int(np.argmax(lev))
        chosen.append(j)
        u = Ginv @ Phi[j]
        c = 1.0 + Phi[j] @ u
        Ginv -= np.outer(u, u) / c
        lev -= (Phi @ u) ** 2 / c
    return np.array(chosen)


def first_step(P, sur, test) -> dict:
    """The ungated one-step errors and the refusals at the test trajectories' first step."""
    X, Y = test[:, 0], test[:, 1]
    with np.errstate(all="ignore"):
        e = np.max(np.where(P.scored(Y), np.abs(X + sur.predict(X) - Y), 0.0), axis=1)
    e = np.where(np.isnan(e), np.inf, e)
    return {"first_step_worst_ungated": float(np.max(e)), "first_step_median_ungated": float(np.median(e)),
            "first_step_refused_percent": float(100 * np.mean(sur.refuses(X)))}  # fmt: skip


def case(name: str, arm: str, store: RecordStore) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    extra, chosen = None, None
    if arm == "m0" and (ref := m0_record(name, None)) is not None:  # R4c's run is this arm: only its first-step scores are new
        return {"from": "r4c", "served": ref.get("served"), "extra": 0, **first_step(P, rebuild(P, train, ref), test)}
    if arm != "m0":
        X0, X1 = extra_pairs(name, P, POOL)
        if arm == "all1600":
            chosen = np.arange(len(X0))
        elif arm.startswith("random400"):
            chosen = np.sort(np.random.default_rng(int(arm.rsplit("_s", 1)[1])).choice(len(X0), 400, replace=False))
        else:
            ref = m0_record(name, store)
            if ref is None:
                raise RuntimeError("leverage400 needs the m0 configuration")
            chosen = leverage(P, train, ref, X0, 400)
        extra = (X0[chosen], X1[chosen])
    out = run(P, train, test, extra=extra, solve_us=baseline_cost(name)["solve_us"])
    sur = rebuild(P, train, out, extra)
    out.update(first_step(P, sur, test))
    out["extra"] = 0 if extra is None else int(len(extra[0]))
    if chosen is not None:
        out["pool_indices"] = chosen.tolist()
    log(f"  {name} {arm}: first step ungated worst {out['first_step_worst_ungated']:.2e}, refused {out['first_step_refused_percent']:.0f}%")
    return out


def main() -> None:
    names = sys.argv[1:] or [*LATTICES, *CHEMISTRY]
    store = RecordStore(OUT / "records" / "r6", path_of=lambda key: OUT / "records" / "r6" / f"{key['problem']}_{key['arm']}.json")
    for name in names:
        for arm in arms(name):
            recorded(store, {"problem": name, "arm": arm, "tol": 0.01}, case, name, arm, store, limit=1800)
    fields = ("status", "extra", "k", "degree", "served", "first_step_worst_ungated", "first_step_median_ungated", "first_step_refused_percent",
              "refused_percent", "rollout_worst", "rollout_solver_share", "mixing_chained", "speedup_effective", "train_s")  # fmt: skip
    summary = {f"{r['key']['problem']}/{r['key']['arm']}": {k: r.get(k) for k in fields} for r in store.records()}
    (OUT / "r6_design.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
