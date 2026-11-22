"""
Descriptors of every step-map problem of the benchmark (memo section P), computed from its cached trajectories and dynamics, and the
Lebesgue function of Proposition 5.1 for chart fits of every admissible degree.

Per problem (tolerance 0.01 in the problem's units):
  dimension   the number of principal directions whose residual max |z - mu - V_m V_m^T (z - mu)| over the states is within half the
              tolerance, for the step-0 states, the states of the last half of the steps, and all visited states (free components; the
              recipe's floor clip); "collapse" when the late count is below the step-0 count;
  dynamics    from the benchmark's dynamics.json: stiffness index, Lyapunov exponent, accumulation A_n at the horizon, kind;
  sampling    for the chart of R2's k (theory/recipe) and every degree q with at least 1.5 pairs per term: the training residual max |G(u_i) -
              y_i|, the Lebesgue function Lambda_X(t) = sum_i |l_i(t)| over the test inputs (median and maximum; no test value used), and the
              one-step error on the test pairs, so Proposition 5.1's two factors can be set beside the error they bound.
Post hoc with respect to R2 (whose fits it re-forms): a description, not a pre-registered test. One record per problem under
data/records/p1/. Usage: p1_descriptors.py [problem ...].
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recipe"))
from common import Chart, ChartPoly, Setup, log, n_terms, recorded  # noqa: E402

from src import PROJECT_DIR  # noqa: E402
from src.benchmark.step_maps import STEP_MAPS, data, visited  # noqa: E402
from src.infra.records import RecordStore  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
R2 = Path(__file__).resolve().parents[1] / "recipe" / "data" / "records" / "r2"
TOL = 0.01


def pca_dimension(Z: np.ndarray, tol: float) -> int:
    mu = Z.mean(axis=0)
    V = np.linalg.svd(Z - mu, full_matrices=False)[2].T
    for m in range(Z.shape[1] + 1):
        R = (Z - mu) - (Z - mu) @ V[:, :m] @ V[:, :m].T
        if np.max(np.abs(R)) <= tol:
            return m
    return Z.shape[1]


def lebesgue(design_train: np.ndarray, design_target: np.ndarray) -> np.ndarray:
    """Lambda_X(t) for every target row: || V (V^T V)^-1 phi(t) ||_1, through a thin QR of the training design V."""
    Q, R = np.linalg.qr(design_train)
    W = np.linalg.solve(R.T, design_target.T)
    return np.abs(Q @ W).sum(axis=0)


def describe(name: str) -> dict:
    P = STEP_MAPS[name]
    train, test = data(name)
    S = Setup(P, train)
    f = S.free
    clip = S.clip
    out = {"d": int(train.shape[2]), "held": int(S.h), "steps": int(train.shape[1] - 1), "n_train": int(len(train)),
           "dimension_step0": pca_dimension(clip(train[:, 0])[:, :f], TOL / 2),
           "dimension_late": pca_dimension(clip(train[:, train.shape[1] // 2 :].reshape(-1, train.shape[2]))[:, :f], TOL / 2),
           "dimension_all": pca_dimension(clip(visited(train))[:, :f], TOL / 2)}  # fmt: skip
    out["collapse"] = bool(out["dimension_late"] < out["dimension_step0"])
    dyn = PROJECT_DIR / "results" / "benchmark" / "data" / "step_maps" / name / "dynamics.json"
    if dyn.exists():
        d = json.loads(dyn.read_text())
        out.update(stiffness_index=d.get("stiffness_index"), lyapunov=d.get("lyapunov"), kind=d.get("kind"),
                   accumulation_horizon=(d.get("accumulation") or [None])[-1])  # fmt: skip
    r2 = json.loads((R2 / f"{name}.json").read_text()) if (R2 / f"{name}.json").exists() else {}
    k = r2.get("k")
    if k is None:
        return out
    chart = Chart(S, k, visited(train))
    U = chart.u(visited(train))
    Xe = test[:, :-1].reshape(-1, test.shape[2])
    Ye = clip(test[:, 1:].reshape(-1, test.shape[2]))[:, :f]
    Ue = chart.u(Xe)
    mask = P.scored(test[:, 1:].reshape(-1, test.shape[2]))[:, :f]
    rows = []
    for q in range(1, 13):
        if n_terms(U.shape[1], q) * 1.5 > len(U) or n_terms(U.shape[1], q) > 3000:
            break
        G = ChartPoly(q, chart.lo, chart.hi).fit(U, S.Y)
        lam = lebesgue(G.design(U), G.design(Ue))
        resid = float(np.max(np.abs(G(U) - S.Y)))
        err = float(np.max(np.where(mask, np.abs(np.clip(G(Ue), S.zlo[:f], S.zhi[:f]) - Ye), 0.0)))
        rows.append({"degree": q, "terms": n_terms(U.shape[1], q), "train_residual": resid, "lebesgue_test_median": float(np.median(lam)),
                     "lebesgue_test_max": float(lam.max()), "test_one_step": err})  # fmt: skip
        log(f"  q={q}: residual {resid:.2e}, Lambda median {np.median(lam):.2e} max {lam.max():.2e}, test one-step {err:.2e}")
    out.update(k=k, degrees=rows)
    return out


def main() -> None:
    names = sys.argv[1:] or sorted(STEP_MAPS, key=lambda n: (data(n)[0].shape[2], n))
    store = RecordStore(OUT / "records" / "p1", path_of=lambda key: OUT / "records" / "p1" / f"{key['problem']}.json")
    for name in names:
        recorded(store, {"problem": name, "method": "descriptors", "tol": TOL}, describe, name, limit=180)
    summary = {r["key"]["problem"]: {k: v for k, v in r.items() if k not in ("key", "provenance", "schema", "recorded")} for r in store.records()}
    (OUT / "p1_descriptors.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")


if __name__ == "__main__":
    main()
