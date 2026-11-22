"""
Corollary 5.3 measured on every problem the generic recipe ran (r4_generic.py's records): the accuracy, refusal and speed trade-off of
the selected surrogate, in the benchmark's field names.

For each problem the selected surrogate is rebuilt exactly as r4 built it (generic.build on all training trajectories with the recorded
k, degree, ridge and held-out error) and evaluated on three sets of inputs: the test states, and the test states mixed once with a random
other test trajectory's state at the same step (w = 0.01 and 0.1, as in the mixing rollouts). For each set and each threshold Lambda* on a
log grid (and the rule's own, tol / (2 e_v) - 1):
  refused_percent  the share of inputs whose Lebesgue bound exceeds Lambda*;
  worst, above_percent  the worst scored one-step error over the accepted inputs and the share of them beyond the tolerance;
  bound            (1 + Lambda*) e_v, Corollary 5.3 (b) with e_v standing in for the best approximation error.
Costs per input, measured on the test states (one core, the run's thread settings): predict_us (surrogate and projection), gate_us (the
Lebesgue bound), and the problem's baseline solve_us (benchmark data); speedup_nominal = solve_us / predict_us (the benchmark's
`speedup`), speedup_effective = solve_us / (gate_us + (1 - rho) predict_us + rho solve_us) at the rule's threshold.

Predictions, committed before the run:
  T1  on every problem the recipe serves with a solver share below 50%, the accepted worst error is within the tolerance at the rule's
      threshold on the test states, and does not increase as Lambda* decreases along the grid (up to 10% sampling noise);
  T2  mixing raises the refusal: at w = 0.1 the refused_percent exceeds the test states' on every such problem;
  T3  the effective speed-up is at most 1 / rho (Corollary 5.3 (c)) on every problem, and where rho exceeds 5% it is below a third of
      the nominal speed-up.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from generic import TOL, build, log
from r4_generic import OUT, kida

from src.benchmark.step_maps import STEP_MAPS, baseline_cost, data
from src.infra.records import RecordStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import recorded  # noqa: E402

GRID = np.logspace(-1, 4, 21)


def mixed(P, test, w, rng):
    """Every test input moved once by mixing with a random other test trajectory's state at the same step (held inputs kept)."""
    h = P.system.held
    free = test.shape[2] - h
    X = test[:, :-1]
    part = np.array([[rng.choice([j for j in range(len(test)) if j != i]) for _ in range(X.shape[1])] for i in range(len(test))])
    Xp = X[part, np.arange(X.shape[1])[None, :]]
    out = X.copy()
    if P.system.log:
        out[..., :free] = np.log10(np.maximum((1 - w) * 10.0 ** X[..., :free] + w * 10.0 ** Xp[..., :free], 1e-300))
    else:
        out[..., :free] = (1 - w) * X[..., :free] + w * Xp[..., :free]
    return out.reshape(-1, test.shape[2])


def curve(P, sur, Z, Ynext, lam_rule):
    with np.errstate(all="ignore"):
        lam = sur.G.lebesgue_bound(sur.T.chart(Z, sur.k))
        err = np.max(np.where(P.scored(Ynext), np.abs(Z + sur.predict(Z) - Ynext), 0.0), axis=1)
    rows = []
    for L in [*GRID, lam_rule]:
        acc = np.isfinite(lam) & (lam <= L) & np.isfinite(err)
        e = err[acc]
        rows.append({"lambda_star": float(L), "refused_percent": float(100 * (1 - acc.mean())), "worst": float(e.max()) if len(e) else 0.0,
                     "above_percent": float(100 * np.mean(e > TOL)) if len(e) else 0.0, "bound": float((1 + L) * sur.e_v)})  # fmt: skip
    return rows[:-1], rows[-1]


def case(name: str, rec: dict) -> dict:
    if name.startswith("kida_"):
        P, train, test = kida(name[5:])
    else:
        P, (train, test) = STEP_MAPS[name], data(name)
    e_v = max(c["one_step_heldout"] for c in rec["selection"] if c["degree"] == rec["degree"] and c["ridge"] == rec["ridge"])
    sur, _ = build(P, train, rec["k"], rec["degree"], rec["ridge"], float(e_v))
    sur.e_v = e_v
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    out = {"lambda_rule": float(sur.lam_star), "e_v": float(e_v), "sets": {}}
    rng = np.random.default_rng(5)
    for label, Z in (("test", Xe), ("mixed_0.01", mixed(P, test, 0.01, rng)), ("mixed_0.1", mixed(P, test, 0.1, rng))):
        Yn = Ye if label == "test" else P.step(Z)
        rows, at_rule = curve(P, sur, Z, Yn, sur.lam_star)
        out["sets"][label] = {"curve": rows, "at_rule": at_rule}
        log(f"  {label}: at the rule's Lambda* {sur.lam_star:.3g}: refused {at_rule['refused_percent']:.1f}%, accepted worst {at_rule['worst']:.2e}")
    n = min(len(Xe), 2000)
    t0 = time.perf_counter()
    sur.predict(Xe[:n])
    predict_us = 1e6 * (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    sur.G.lebesgue_bound(sur.T.chart(Xe[:n], sur.k))
    gate_us = 1e6 * (time.perf_counter() - t0) / n
    solve_us = baseline_cost(name)["solve_us"] if not name.startswith("kida_") else None
    rho = out["sets"]["test"]["at_rule"]["refused_percent"] / 100
    out.update(predict_us=predict_us, gate_us=gate_us, solve_us=solve_us)
    if solve_us:
        out["speedup_nominal"] = solve_us / predict_us
        out["speedup_effective"] = solve_us / (gate_us + (1 - rho) * predict_us + rho * solve_us)
        log(f"  predict {predict_us:.1f} us, gate {gate_us:.1f} us, solve {solve_us:.0f} us: speed-up nominal {out['speedup_nominal']:.1f}, effective {out['speedup_effective']:.1f} (refused {100 * rho:.1f}%)")
    return out


def main() -> None:
    r4 = RecordStore(OUT / "records" / "r4")
    store = RecordStore(OUT / "records" / "p2", path_of=lambda key: OUT / "records" / "p2" / f"{key['problem']}.json")
    names = sys.argv[1:] or [r["key"]["problem"] for r in r4.records() if r.get("status") == "done"]
    for name in names:
        rec = json.loads((OUT / "records" / "r4" / f"{name}.json").read_text())
        if rec.get("status") != "done":
            continue
        recorded(store, {"problem": name, "method": "tradeoff", "tol": TOL}, case, name, rec, limit=600)
    summary = {r["key"]["problem"]: {k: v for k, v in r.items() if k in ("lambda_rule", "e_v", "predict_us", "gate_us", "solve_us",
               "speedup_nominal", "speedup_effective")} | {s: v["at_rule"] for s, v in r.get("sets", {}).items()} for r in store.records()}  # fmt: skip
    (OUT / "p2_tradeoff.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
