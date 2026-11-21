"""
Function-suite pass, aggregates (memo T7's open direct test): lotka_volterra at d = 20 on [0.1, 1]^20, the first species of a
competitive Lotka-Volterra network after t = 2, coupled to every species through sum_j A_1j x_j.

Prediction (T7): the map is driven by a few aggregate directions; a low-degree polynomial in the top k of them is accurate on
most of the box (the uniform measure concentrates the aggregates) and much worse at the box's vertices, where the aggregates
reach their extremes. So a measure-sense target needs few coordinates and a low degree; the box's sup norm needs more.

Measured: the gradient covariance C = E[grad f grad f^T] from 300 uniform points (central differences), its eigenvalues; for
k = 1, 2, 3 and total degree q = 2, 3, 4, a least-squares polynomial in the top-k eigen-coordinates fitted on 6,000 uniform
points; its error (relative to the range of f) on 20,000 fresh uniform points (worst and 99.9% quantile) and on 2,000 random
vertices (worst).
"""

import itertools
import json
from pathlib import Path

import numpy as np

from src.problems.registry import FUNCTIONS

OUT = Path(__file__).resolve().parent / "data"
D, LO, HI = 20, 0.1, 1.0


def f(X):
    return np.asarray(FUNCTIONS["lotka_volterra"].f(X)).reshape(len(X))


def features(Y, q):
    k = Y.shape[1]
    cols = [np.ones(len(Y))]
    for deg in range(1, q + 1):
        for idx in itertools.combinations_with_replacement(range(k), deg):
            cols.append(np.prod(Y[:, list(idx)], axis=1))
    return np.stack(cols, axis=1)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    G = rng.uniform(LO, HI, (300, D))
    h = 1e-5
    grads = np.stack([(f(G + h * e) - f(G - h * e)) / (2 * h) for e in np.eye(D)], axis=1)
    C = grads.T @ grads / len(G)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    Xtr, Xte = rng.uniform(LO, HI, (6000, D)), rng.uniform(LO, HI, (20000, D))
    Xv = np.where(rng.random((2000, D)) < 0.5, LO, HI)
    ftr, fte, fv = f(Xtr), f(Xte), f(Xv)
    scale = float(max(fte.max(), fv.max()) - min(fte.min(), fv.min()))
    c = 0.5 * (LO + HI) * np.ones(D)
    rows = []
    for k in (1, 2, 3):
        P = V[:, :k]
        for q in (2, 3, 4):
            A = features((Xtr - c) @ P, q)
            coef = np.linalg.lstsq(A, ftr, rcond=None)[0]
            et = np.abs(features((Xte - c) @ P, q) @ coef - fte) / scale
            ev = np.abs(features((Xv - c) @ P, q) @ coef - fv) / scale
            rows.append({"k": k, "degree": q, "terms": int(A.shape[1]), "uniform_worst": float(et.max()), "uniform_q999": float(np.quantile(et, 0.999)),
                         "vertex_worst": float(ev.max()), "vertex_over_uniform": float(ev.max() / et.max())})  # fmt: skip
    out = {"d": D, "eigenvalues_normalised": (lam / lam[0]).tolist(), "range_of_f": scale, "rows": rows}
    (OUT / "f4_aggregates.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("gradient covariance eigenvalues / largest:", np.round(lam[:6] / lam[0], 4).tolist(), " range of f", round(scale, 4))
    for r in rows:
        print(f"k={r['k']} q={r['degree']} ({r['terms']} terms): uniform worst {r['uniform_worst']:.2e} (99.9% {r['uniform_q999']:.2e}), vertices worst {r['vertex_worst']:.2e}, ratio {r['vertex_over_uniform']:.1f}")


if __name__ == "__main__":
    main()
