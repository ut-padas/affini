"""
Test 1 (HIRES, k = 3, tolerance 0.01 dex): what the 9,477 solver calls of Demonstration 8.5 buy.
(a) data-only fits: a least-squares polynomial from the chart value to the next state on the training pairs (no solver call beyond the
    trajectories every method is given), projected onto the invariant, scored like the benchmark (all test states; 20 rollouts);
(b) the ideal reduced map Phi(D(pi z)) (the function ChebTT approximates; the lift error alone), one step and rollout;
(c) how much of the chart box the visited states occupy, and how far the decoder leaves the data's range on the rest.
"""
import json
import time

import numpy as np
from common import HERE, FnModel, poly, setup

from src.benchmark.step_maps import ROLLOUTS, Projected, rollout

s = setup()
p, test, mu, V, lo, hi, W = s["p"], s["test"], s["mu"], s["V"], s["lo"], s["hi"], s["W"]
X, Y, U = s["X"], s["Y"], s["U"]
Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
mask = p.scored(Ye)
c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
out = {}


def score(fn, label):
    m = Projected(FnModel(fn, mu, V, lo, hi), p, W)
    e = np.max(np.where(mask, np.abs(Xe + m.predict(Xe) - Ye), 0.0), axis=1)
    r = rollout(p, m, test[:ROLLOUTS])
    out[label] = {"one_step_worst": float(e.max()), "rollout_worst": r["rollout_worst"], "rollout_end": r["rollout_error_by_step"][-1]}
    print(label, out[label], flush=True)


for q in (3, 4, 5, 6, 7, 8):  # scaled chart coordinates, so the monomial basis is well conditioned
    A = poly((U - c) / h, q)
    coef = np.linalg.lstsq(A, Y, rcond=None)[0]
    score(lambda Uq, coef=coef, q=q: poly((Uq - c) / h, q) @ coef, f"data-only degree {q} ({A.shape[1]} terms, cond {np.linalg.cond(A):.1e})")

t0 = time.perf_counter()
score(lambda Uq: p.step(poly(Uq, s["q"]) @ s["coef"]), "ideal reduced Phi(D(pi z)), all 6800 test states")
print("ideal seconds", time.perf_counter() - t0)

# (c) occupancy of the chart box
rng = np.random.default_rng(0)
Q = lo + (hi - lo) * rng.random((20000, 3))
Un = (U - c) / h
from scipy.spatial import cKDTree

dist = cKDTree(Un).query((Q - c) / h)[0]
gap = cKDTree(Un).query(Un, k=2)[0][:, 1]
D = poly(Q, s["q"]) @ s["coef"]
xlo, xhi = X.min(0), X.max(0)
excess = np.max(np.maximum(xlo - D, D - xhi), axis=1)  # dex outside the data's bounding box
out["box"] = {"share_of_box_within_0.05_of_data": float(np.mean(dist < 0.05)), "share_within_0.1": float(np.mean(dist < 0.1)),
              "median_nn_gap_of_data": float(np.median(gap)), "q99_nn_gap_of_data": float(np.quantile(gap, 0.99)),
              "decoder_outside_data_box_dex_max": float(excess.max()), "decoder_outside_share_gt_0.1dex": float(np.mean(excess > 0.1)),
              "decoder_outside_share_gt_1dex": float(np.mean(excess > 1.0))}
print(out["box"])
(HERE / "t1_baselines.json").write_text(json.dumps(out, indent=1))
