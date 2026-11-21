"""
Test 7 (HIRES): a chart from the step map's sensitive input directions instead of the visited states' principal directions.
C = mean of DPhi^T DPhi over 150 training states (central differences); V_J its top-k eigenvectors; pi_J(z) = V_J^T (z - mu); decoder
D_J of total degree 4 from pi_J to the states, as in curved_lift.py. For k = 3, 4 and for the PCA chart, on 400 held-out visited states:
  normal gain   the largest singular value of DPhi restricted to the chart's normal space (median, max over 40 states): what the exact
                step keeps of a kick the encoder cannot see;
  lift error    max |Phi(D(pi z)) - Phi(z)| (on the visited set);
  kicked error  max |Phi(D(pi (z + delta))) - Phi(z + delta)| for kicks of max-norm 0.01 and 0.05 dex in every component (test 2's
                first row; the ideal reduced map, no ChebTT), beside what the exact step retains of the kick.
"""
import json

import numpy as np
from common import HERE, poly, setup

s = setup()
p, test, X, mu = s["p"], s["test"], s["X"], s["mu"]
Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
E = np.zeros((16, 8))
for i in range(8):
    E[2 * i, i], E[2 * i + 1, i] = 1e-5, -1e-5


def jac(z):
    o = p.step(z[None] + E)
    return (o[0::2] - o[1::2]).T / 2e-5


rng = np.random.default_rng(0)
C = np.mean([J.T @ J for J in (jac(z) for z in X[rng.choice(len(X), 150, replace=False)])], axis=0)
w, Q = np.linalg.eigh(C)
print("eigenvalues of mean DPhi^T DPhi / largest:", np.round(w[::-1] / w[-1], 5).tolist())
idx = np.random.default_rng(0).choice(len(Xe), 400, replace=False)
Z, Zn = Xe[idx], Ye[idx]
Jt = [jac(z) for z in Xe[np.random.default_rng(3).choice(len(Xe), 40, replace=False)]]
kick = np.random.default_rng(1).choice([-1.0, 1.0], Z.shape)
pca = np.linalg.svd(X - mu, full_matrices=False)[2]
out = {"eigenvalues": (w[::-1] / w[-1]).tolist(), "rows": []}
for label, V in (("PCA k=3", pca[:3].T), ("PCA k=4", pca[:4].T), ("Jacobian k=3", Q[:, ::-1][:, :3]), ("Jacobian k=4", Q[:, ::-1][:, :4])):
    k = V.shape[1]
    N = np.linalg.svd(V.T)[2][k:].T
    gains = [np.linalg.svd(J @ N, compute_uv=False)[0] for J in Jt]
    coef = np.linalg.lstsq(poly((X - mu) @ V, 4), X, rcond=None)[0]
    psi = lambda Zq: p.step(poly((Zq - mu) @ V, 4) @ coef)  # noqa: E731
    row = {"chart": label, "normal_gain_median": float(np.median(gains)), "normal_gain_max": float(np.max(gains)),
           "decoder_residual": float(np.max(np.abs(poly((Z - mu) @ V, 4) @ coef - Z))),
           "lift_error": float(np.max(np.where(p.scored(Zn), np.abs(psi(Z) - Zn), 0.0)))}
    for eps in (0.01, 0.05):
        Zk = Z + eps * kick
        true = p.step(Zk)
        m = p.scored(true)
        row[f"retained_{eps}"] = float(np.max(np.where(m, np.abs(true - Zn), 0.0)))
        row[f"kicked_error_{eps}"] = float(np.max(np.where(m, np.abs(psi(Zk) - true), 0.0)))
    out["rows"].append(row)
    print({a: (round(b, 5) if isinstance(b, float) else b) for a, b in row.items()}, flush=True)
(HERE / "t7_jacobian_chart.json").write_text(json.dumps(out, indent=1))
