"""
Test 4 (POLLU, tolerance 0.01 dex): is "the decoder failed, so POLLU needs an atlas" (Demonstration 8.5) a property of the visited set
or of one decoder (k = 5, total degree 4, unscaled monomials)?
(a) the visited set's dimension: singular values of the 100 training states at one time step (the 6 varied emissions; time adds one),
    and the local singular values on neighbourhoods of 60 states with the number of distinct trajectories each neighbourhood holds;
(b) decoders for k = 5..9: affine, least-squares polynomials of degree 2-4 in scaled chart coordinates (minimum-norm solution), and a
    local affine fit on the 40 nearest training chart values (a graph test without a global polynomial): the decoder's residual and the
    lift error max |Phi(lift(pi z)) - Phi(z)| on 400 held-out visited states.
"""
import json
import time

import numpy as np
from common import HERE, poly, setup
from scipy.spatial import cKDTree

out = {}
s = setup("pollu", k=20)
p, train, test, X, mu = s["p"], s["train"], s["test"], s["X"], s["mu"]
Vt = np.linalg.svd(X - mu, full_matrices=False)[2]
d = X.shape[1]
# (a)
for step in (0, 10, 30, 58):
    S = train[:, step]
    sv = np.linalg.svd(S - S.mean(0), compute_uv=False)
    print(f"step {step}: singular values of the 100 states / first", np.round(sv[:9] / sv[0], 4).tolist())
    out[f"cross_section_step_{step}"] = (sv / sv[0]).tolist()[:10]
tree = cKDTree(X)
rng = np.random.default_rng(0)
pick = rng.choice(len(X), 300, replace=False)
traj_of = np.tile(np.arange(train.shape[0]), train.shape[1] - 1)  # time-major order
loc, ntraj = [], []
for i in pick:
    nb = tree.query(X[i], k=60)[1]
    sv = np.linalg.svd(X[nb] - X[nb].mean(0), compute_uv=False)
    loc.append(sv[:8] / sv[0])
    ntraj.append(len(set(traj_of[nb])))
out["local_sv_median"] = np.median(loc, axis=0).tolist()
out["local_distinct_trajectories_median"] = float(np.median(ntraj))
print("local singular values (median, 60 neighbours):", np.round(out["local_sv_median"], 4).tolist(), "; distinct trajectories per neighbourhood: median",
      np.median(ntraj), "min", min(ntraj), "max", max(ntraj))
# (b)
Xe, Ye = test[:, :-1].reshape(-1, d), test[:, 1:].reshape(-1, d)
idx = np.random.default_rng(0).choice(len(Xe), 400, replace=False)
Ze, Zn = Xe[idx], Ye[idx]
mask = p.scored(Zn)
rows = []
t0 = time.perf_counter()
import sys
KS = tuple(int(a) for a in sys.argv[1:]) or (5, 6)
for k in KS:
    V = Vt[:k].T
    U, Ue = (X - mu) @ V, (Ze - mu) @ V
    c, h = 0.5 * (U.min(0) + U.max(0)), 0.5 * (U.max(0) - U.min(0))
    lifts = {"affine": mu + Ue @ V.T}
    for q in (2, 3, 4):
        A = poly((U - c) / h, q)
        if A.shape[1] > 800:
            continue
        coef = np.linalg.lstsq(A, X, rcond=1e-10)[0]
        lifts[f"degree {q} ({A.shape[1]} terms)"] = poly((Ue - c) / h, q) @ coef
    kt = cKDTree(U / h)
    nb = kt.query(Ue / h, k=40)[1]
    L = np.empty_like(Ze)
    for j in range(len(Ue)):
        A = np.hstack([np.ones((40, 1)), (U[nb[j]] - Ue[j]) / h])
        L[j] = np.linalg.lstsq(A, X[nb[j]], rcond=1e-8)[0][0]
    lifts["local affine, 40 neighbours"] = L
    for name, Zl in lifts.items():
        resid = float(np.max(np.abs(Zl - Ze)))
        lift = float(np.max(np.where(mask, np.abs(p.step(Zl) - Zn), 0.0)))
        rows.append({"k": k, "decoder": name, "residual": resid, "lift_error": lift})
        print(f"k={k} {name:28s} residual {resid:.2e} lift error {lift:.2e}  ({time.perf_counter() - t0:.0f} s)", flush=True)
out["decoders"] = rows
(HERE / f"t4_pollu_k{KS[0]}-{KS[-1]}.json").write_text(json.dumps(out, indent=1))
