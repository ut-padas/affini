"""Refutation test 3: genz_continuous at d = 3 (ChebTT with breakpoints: 2 pieces per axis, rank 1, evaluations flat in the tolerance), and what ChebTT without breakpoints does at relative tolerances 0.3 to 0.03 at d = 2, where the n^-1 law predicts degrees of only about 8 to 64 per axis."""
import sys
import time

import numpy as np

from src.problems.registry import FUNCTIONS
from src.schemes import chebtt

prob = FUNCTIONS["genz_continuous"]
f = lambda X: np.asarray(prob.f(X)).reshape(len(X))  # noqa: E731
for d in (3, 4):
    scale = float(np.abs(f(np.random.default_rng(0).uniform(0, 1, (20000, d)))).max())
    for tol in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        m = chebtt.build(f, np.zeros(d), np.ones(d), tol * scale, seed=1, max_evals=400_000, edges=True)
        print(f"edges d={d} tol {tol:.0e}: degrees {list(map(int, m.degrees))} pieces {[m.pieces(k) for k in range(d)]} ranks {list(map(int, m.ranks))} evals {int(m.evals)}")
d = 2
scale = float(np.abs(f(np.random.default_rng(0).uniform(0, 1, (20000, d)))).max())
print("scale (max |f| on 20,000 uniform points) at d=2:", scale)
for tol in (0.3, 0.1, 0.03):
    for cap in (20_000, 200_000):
        t0 = time.time()
        try:
            m = chebtt.build(f, np.zeros(d), np.ones(d), tol * scale, seed=1, max_evals=cap, edges=False)
            res = f"degrees {list(map(int, m.degrees))} ranks {list(map(int, m.ranks))} evals {int(m.evals)}"
        except Exception as e:  # noqa: BLE001
            res = f"{type(e).__name__}: {str(e)[:60]}"
        print(f"no edges d=2 tol {tol} max_evals {cap}: {res}  [{time.time() - t0:.1f} s]")
        sys.stdout.flush()
