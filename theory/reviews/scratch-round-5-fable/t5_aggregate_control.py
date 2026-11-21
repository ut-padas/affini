"""
Control for section F's aggregate row (f4_aggregates.py): is "4-12 times worse at the vertices" specific to a few aggregate
coordinates, or what any least-squares polynomial fitted on uniform points does at the vertices (extrapolation)?
Arms: (A) the memo's fit: top-k gradient-covariance directions, degree q, trained on uniform points;
      (B) a full quadratic in all 20 coordinates (231 terms), trained on the same uniform points: no coordinate is dropped;
      (C) arm A trained on uniform points plus 2,000 other random vertices: the same few coordinates, vertices seen.
"""
import itertools

import numpy as np

from src.problems.registry import FUNCTIONS

D, LO, HI = 20, 0.1, 1.0
f = lambda X: np.asarray(FUNCTIONS["lotka_volterra"].f(X)).reshape(len(X))  # noqa: E731


def features(Y, q):
    cols = [np.ones(len(Y))]
    for deg in range(1, q + 1):
        for idx in itertools.combinations_with_replacement(range(Y.shape[1]), deg):
            cols.append(np.prod(Y[:, list(idx)], axis=1))
    return np.stack(cols, axis=1)


rng = np.random.default_rng(0)
G = rng.uniform(LO, HI, (300, D))
h = 1e-5
grads = np.stack([(f(G + h * e) - f(G - h * e)) / (2 * h) for e in np.eye(D)], axis=1)
lam, V = np.linalg.eigh(grads.T @ grads / len(G))
V = V[:, ::-1]
Xtr, Xte = rng.uniform(LO, HI, (6000, D)), rng.uniform(LO, HI, (20000, D))
Xv = np.where(rng.random((2000, D)) < 0.5, LO, HI)
Xv2 = np.where(np.random.default_rng(5).random((2000, D)) < 0.5, LO, HI)
ftr, fte, fv, fv2 = f(Xtr), f(Xte), f(Xv), f(Xv2)
scale = float(max(fte.max(), fv.max()) - min(fte.min(), fv.min()))
c = 0.55 * np.ones(D)


def report(label, P, q, Xa, fa):
    A = features((Xa - c) @ P, q)
    coef = np.linalg.lstsq(A, fa, rcond=None)[0]
    et = np.abs(features((Xte - c) @ P, q) @ coef - fte).max() / scale
    ev = np.abs(features((Xv - c) @ P, q) @ coef - fv).max() / scale
    print(f"{label:60s} terms {A.shape[1]:4d}: uniform worst {et:.2e}, vertices worst {ev:.2e}, ratio {ev / et:.1f}")


for k, q in ((1, 4), (2, 3), (3, 3), (3, 4)):
    report(f"A  top-{k} directions, degree {q}, uniform training", V[:, :k], q, Xtr, ftr)
    report(f"C  top-{k} directions, degree {q}, uniform + vertices training", V[:, :k], q, np.vstack([Xtr, Xv2]), np.concatenate([ftr, fv2]))
report("B  all 20 coordinates, degree 1, uniform training", np.eye(D), 1, Xtr, ftr)
report("B  all 20 coordinates, degree 2, uniform training", np.eye(D), 2, Xtr, ftr)
report("B' all 20 coordinates, degree 2, uniform + vertices training", np.eye(D), 2, np.vstack([Xtr, Xv2]), np.concatenate([ftr, fv2]))
# how far is a vertex from the uniform bulk, in the leading aggregate?
s_u, s_v = (Xte - c) @ V[:, 0], (Xv - c) @ V[:, 0]
print("leading aggregate: uniform range", s_u.min().round(3), s_u.max().round(3), " vertex range", s_v.min().round(3), s_v.max().round(3), " box extreme +-", (0.45 * np.abs(V[:, 0]).sum()).round(3))
