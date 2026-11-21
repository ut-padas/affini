"""Refutation test 1: separable analytic functions at d = 3 with known poles; ChebTT's degree per decade against 1/log10(rho_i), and rank 1."""
import math
import sys
import time

import numpy as np

from src.problems.registry import FUNCTIONS
from src.schemes import chebtt
from t1_recompute import params, rho

TOLS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


def run(f, lo, hi, d, label, pred):
    scale = float(np.abs(f(np.random.default_rng(0).uniform(lo, hi, (20000, d)))).max())
    rows = []
    for t in TOLS:
        m = chebtt.build(f, np.full(d, lo), np.full(d, hi), t * scale, seed=1, max_evals=400_000, edges=False)
        rows.append((t, list(map(int, m.degrees)), list(map(int, m.ranks)), int(m.evals)))
    dec = [math.log10(1 / r[0]) for r in rows]
    meas = [float(np.polyfit(dec, [r[1][i] for r in rows], 1)[0]) for i in range(d)]
    print(label)
    for r in rows:
        print("   ", r)
    print("   predicted degree/decade", np.round(pred, 2), "measured", np.round(meas, 2), "ratio", np.round(np.array(meas) / np.array(pred), 2), "ranks", sorted({x for r in rows for x in r[2]}))
    sys.stdout.flush()


t0 = time.time()
# (a) the benchmark's product peak at d = 3 (its own a, w)
a, w = params(7.25, 3)
prob = FUNCTIONS["genz_product_peak"]
run(lambda X: np.asarray(prob.f(X)).reshape(len(X)), 0.0, 1.0, 3, f"genz_product_peak d=3, a={np.round(a, 3)}, w={np.round(w, 3)}", [1 / math.log10(rho(w[i] + 1j / a[i], 0, 1)) for i in range(3)])
# (b) a product of Runge factors on [-1, 1]^3 with poles at +- i b_i: rho_i = b_i + sqrt(1 + b_i^2)
b = np.array([0.2, 0.5, 1.5])
run(lambda X: np.prod(1.0 / (b**2 + X**2), axis=1), -1.0, 1.0, 3, f"prod 1/(b_i^2 + x_i^2), b={b}", [1 / math.log10(bi + math.sqrt(1 + bi * bi)) for bi in b])
# (c) a product with a real pole outside the box: prod 1/(c_i - x_i), rho_i = c_i + sqrt(c_i^2 - 1)
c = np.array([1.05, 1.5, 4.0])
run(lambda X: np.prod(1.0 / (c - X), axis=1), -1.0, 1.0, 3, f"prod 1/(c_i - x_i), c={c}", [1 / math.log10(ci + math.sqrt(ci * ci - 1)) for ci in c])
print("seconds", round(time.time() - t0, 1))
