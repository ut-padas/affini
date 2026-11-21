"""
Test 5 (HIRES, k = 3): (a) the invariant of the rebuilt reduced ChebTT, one step and along the 170-step rollouts; (b) is the visited set
attracting: singular values of the step map's Jacobian at 40 held-out visited states, and of its restriction to the 5 directions normal
to the chart (a slow manifold of the chart's dimension would give normal gains << 1).
"""
import json

import numpy as np
from common import HERE, load_reduced, setup

s = setup()
p, test, mu, V, W = s["p"], s["test"], s["mu"], s["V"], s["W"]
sur = load_reduced(s)
Xe = test[:, :-1].reshape(-1, 8)
Yn, Y0 = 10.0 ** (Xe + sur.predict(Xe)), 10.0**Xe
one = float(np.max(np.abs((Yn - Y0) @ W.T) / np.abs(Y0 @ W.T)))
Z = test[:20, 0].copy()
w0 = (10.0**Z) @ W.T
for _ in range(170):
    Z = Z + sur.predict(Z)
roll = float(np.max(np.abs((10.0**Z) @ W.T - w0) / np.abs(w0)))
print("invariant rows", W.round(3).tolist(), "one-step relative change", one, "after 170 chained steps", roll)
idx = np.random.default_rng(3).choice(len(Xe), 40, replace=False)
N = np.linalg.svd(V.T)[2][3:].T  # (8, 5) orthonormal basis of the chart's normal space
E = np.zeros((16, 8))
for i in range(8):
    E[2 * i, i], E[2 * i + 1, i] = 1e-5, -1e-5
full, normal = [], []
for z in Xe[idx]:
    o = p.step(z[None] + E)
    J = (o[0::2] - o[1::2]).T / 2e-5
    full.append(np.linalg.svd(J, compute_uv=False))
    normal.append(np.linalg.svd(J @ N, compute_uv=False))
out = {"invariant_one_step": one, "invariant_170_steps": roll, "J_singular_values_median": np.median(full, 0).tolist(),
       "J_normal_singular_values_median": np.median(normal, 0).tolist(), "J_normal_singular_values_min_over_states": np.min(normal, 0).tolist()}
print("singular values of DPhi (median over 40 states):", np.round(out["J_singular_values_median"], 4).tolist())
print("of DPhi restricted to the chart's normal space :", np.round(out["J_normal_singular_values_median"], 4).tolist())
(HERE / "t5_checks.json").write_text(json.dumps(out, indent=1))
