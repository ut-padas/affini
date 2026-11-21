"""
Refutation test 2: a general (non-trigonometric) oblique ridge g(q.x) on [-1, 1]^d, q a random unit vector.
Section F's corrected claim: ChebTT's cost is set by the profile's TT rank, and "a general profile g(q.x) would have growing rank".
Lemma 7.2 predicts more: the rank is at most (degree of g in s at that tolerance) + 1, whatever d, so it grows with the tolerance
(about 1/log10(rho_s) per decade for an analytic g) and NOT with d, and ChebTT's evaluations grow about linearly in d.
Profiles: Runge g(s) = 1/(1 + (s/b)^2) (poles at +- i b), and exp(s) (entire).
"""
import math
import sys
import time

import numpy as np

from src.schemes import chebtt
from t1_recompute import rho

t0 = time.time()
for name, g, pole in (("runge b=0.7", lambda s: 1 / (1 + (s / 0.7) ** 2), 0.7j), ("runge b=0.3", lambda s: 1 / (1 + (s / 0.3) ** 2), 0.3j), ("exp(2s)", lambda s: np.exp(2 * s), None)):
    for d in (2, 3, 5, 8):
        q = np.random.default_rng(100 + d).normal(size=d)
        q /= np.linalg.norm(q)
        S = float(np.abs(q).sum())  # s ranges over [-S, S]
        f = lambda X, q=q, g=g: g(X @ q)  # noqa: E731
        out = []
        for tol in (1e-2, 1e-4, 1e-6, 1e-8):
            if time.time() - t0 > 140:
                break
            try:
                m = chebtt.build(f, -np.ones(d), np.ones(d), tol, seed=1, max_evals=300_000, edges=False)
                out.append((max(map(int, m.ranks)), int(m.evals)))
            except RuntimeError as e:
                out.append(("fail", str(e)[:30]))
        pred = "entire" if pole is None else f"rank/decade <= {1 / math.log10(rho(pole, -S, S)):.2f}"
        ranks = [o[0] for o in out]
        sl = np.polyfit([2, 4, 6, 8][: len(ranks)], ranks, 1)[0] if all(isinstance(r, int) for r in ranks) and len(ranks) > 1 else float("nan")
        print(f"{name:12s} d={d} |q|_1={S:.2f} {pred}: (max rank, evals) at 1e-2,1e-4,1e-6,1e-8: {out}; measured rank/decade {sl:.2f}")
        sys.stdout.flush()
print("seconds", round(time.time() - t0, 1))
