"""
Test 9 (HIRES, k = 3): error tracking off the visited set for the reduced ChebTT of Demonstration 8.5.
(a) an indicator that costs no solve: the decoder residual r(z) = max |z - D(pi z)|. On 400 held-out states kicked by eps in every
    component, the surrogate's error against r: the largest error / (e0 + r) with e0 the on-set worst, and what a refusal at r > tol / 4
    would leave.
(b) a first-order normal correction, Psi1(z) = Phi(D(pi z)) + DPhi(D(pi z)) (z - D(pi z)) (exact Jacobian by central differences, 100
    states): the error left, against the uncorrected Phi(D(pi z)).
"""
import json

import numpy as np
from common import HERE, load_reduced, poly, setup

s = setup()
p, test, mu, V = s["p"], s["test"], s["mu"], s["V"]
sur = load_reduced(s)
Xe = test[:, :-1].reshape(-1, 8)
idx = np.random.default_rng(0).choice(len(Xe), 400, replace=False)
Z = Xe[idx]
kick = np.random.default_rng(1).choice([-1.0, 1.0], Z.shape)
dec = lambda Zq: poly((Zq - mu) @ V, s["q"]) @ s["coef"]  # noqa: E731
E = np.zeros((16, 8))
for i in range(8):
    E[2 * i, i], E[2 * i + 1, i] = 1e-5, -1e-5
out = []
for eps in (0.0, 1e-3, 3e-3, 1e-2, 5e-2):
    Zk = Z + eps * kick
    true = p.step(Zk)
    m = p.scored(true)
    err = np.max(np.where(m, np.abs(Zk + sur.predict(Zk) - true), 0.0), axis=1)
    r = np.max(np.abs(Zk - dec(Zk)), axis=1)
    keep = r <= 0.0025
    row = {"eps": eps, "error_worst": float(err.max()), "residual_median": float(np.median(r)), "error_over_residual_max": float(np.max(err / (0.00246 + r))),
           "error_over_residual_median": float(np.median(err / (0.00246 + r))), "kept_percent_at_r_le_tol/4": float(100 * keep.mean()),
           "error_worst_kept": float(err[keep].max()) if keep.any() else None}
    L = dec(Zk[:100])
    base = p.step(L)
    corr = base.copy()
    for j in range(100):
        o = p.step(L[j][None] + E)
        corr[j] += ((o[0::2] - o[1::2]).T / 2e-5) @ (Zk[j] - L[j])
    row["ideal_uncorrected_worst_100"] = float(np.max(np.where(m[:100], np.abs(base - true[:100]), 0.0)))
    row["ideal_first_order_corrected_worst_100"] = float(np.max(np.where(m[:100], np.abs(corr - true[:100]), 0.0)))
    out.append(row)
    print({a: (round(b, 5) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
(HERE / "t9_tracking.json").write_text(json.dumps(out, indent=1))
