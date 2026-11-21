"""
Test 8 (HIRES): the affine lift of Demonstration 8.4 in a Jacobian chart (top-k eigenvectors of the mean DPhi^T DPhi over 150 training
states) against the PCA chart, on 400 held-out visited states and on the same states kicked by 0.01 and 0.05 dex in every component:
max |Phi(mu + V V^T (z - mu)) - Phi(z)|. No decoder, so nothing is extrapolated off the visited set.
"""
import json

import numpy as np
from common import HERE, setup

s = setup()
p, test, X, mu = s["p"], s["test"], s["X"], s["mu"]
Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
E = np.zeros((16, 8))
for i in range(8):
    E[2 * i, i], E[2 * i + 1, i] = 1e-5, -1e-5
rng = np.random.default_rng(0)
C = np.zeros((8, 8))
for z in X[rng.choice(len(X), 150, replace=False)]:
    o = p.step(z[None] + E)
    J = (o[0::2] - o[1::2]).T / 2e-5
    C += J.T @ J
Q = np.linalg.eigh(C)[1][:, ::-1]
pca = np.linalg.svd(X - mu, full_matrices=False)[2].T
idx = np.random.default_rng(0).choice(len(Xe), 400, replace=False)
Z = Xe[idx]
kick = np.random.default_rng(1).choice([-1.0, 1.0], Z.shape)
rows = []
for eps in (0.0, 0.01, 0.05):
    Zk = Z + eps * kick
    true = p.step(Zk)
    m = p.scored(true)
    for k in range(3, 8):
        row = {"eps": eps, "k": k}
        for label, B in (("pca", pca), ("jacobian", Q)):
            V = B[:, :k]
            row[label] = float(np.max(np.where(m, np.abs(p.step(mu + (Zk - mu) @ V @ V.T) - true), 0.0)))
        rows.append(row)
        print(row, flush=True)
(HERE / "t8_jacobian_affine.json").write_text(json.dumps(rows, indent=1))
