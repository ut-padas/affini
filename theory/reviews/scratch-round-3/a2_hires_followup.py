"""
Round-3 test A, follow-up: (1) is the divergence of the charts with k = 4, 5 coordinates an artefact of the degree? (k = 4 at
degrees 2, 3, 5); (2) does the k = 3 surrogate's rollout error follow its one-step error times the exact dynamics' own
amplification on the visited set? (degrees 3, 4, 6, 8, against the growth of the difference between two exact trajectories
whose initial parameters differ by 1e-4 relative); (3) the dimension of the visited set: local PCA of the 40 nearest visited
neighbours of 200 visited states.
"""
import json
from pathlib import Path

import numpy as np
from a_hires_chart import DT, STEPS, hires_initial, jacobian, make, step, trajectories

HERE = Path(__file__).resolve().parent
cache = HERE / "hires_trajectories.npz"
if cache.exists():
    train, test = (np.load(cache)[k] for k in ("train", "test"))
else:
    train, test = trajectories(hires_initial(100, 1)), trajectories(hires_initial(40, 2))
    np.savez(cache, train=train, test=test)
X, Y = train[:, :-1].reshape(-1, 8), train[:, 1:].reshape(-1, 8)
Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
rng = np.random.default_rng(0)
mean = X.mean(0)
pca = np.linalg.svd(X - mean, full_matrices=False)[2]
out = {}

# (3) local dimension of the visited set
loc = []
for i in rng.choice(len(X), 200, replace=False):
    nb = X[np.argsort(np.linalg.norm(X - X[i], axis=1))[:40]]
    s = np.linalg.svd(nb - nb.mean(0), compute_uv=False)
    loc.append(s / s[0])
loc = np.median(loc, 0)
out["local_pca_median"] = loc.tolist()
print("local PCA (40 neighbours), median singular values / first:", np.array2string(loc, precision=4))

# (2a) the exact dynamics' amplification on the visited set
Y0 = hires_initial(40, 2)[:12]
Yp = Y0.copy()
Yp[:, 0] *= 1 + 1e-4
Yq = Y0.copy()
Yq[:, 7] *= 1 + 1e-4
base, tp, tq = trajectories(Y0), trajectories(Yp), trajectories(Yq)
amp = {}
for name, t in (("Pr0", tp), ("E", tq)):
    d = np.abs(t - base).max(2)  # (traj, step)
    a = np.maximum.accumulate(d, axis=1) / d[:, :1]
    amp[name] = {str(m): float(a[:, m].max()) for m in (1, 10, 40, STEPS - 1)}
    print(f"exact amplification of a 1e-4 relative change of {name}: " + " ".join(f"step {m}: {v:.1f}" for m, v in amp[name].items()))
out["exact_amplification"] = amp

# (1), (2b)
rows = []
print(f"{'chart':8s} {'deg':>3s} {'feat':>4s} {'1-step':>9s} | rollout at 1, 10, 40, 169 | max rho(DPsi)")
for k, q in ((3, 3), (3, 4), (3, 6), (3, 8), (4, 2), (4, 3), (4, 5), (2, 12)):
    chart = lambda Z, k=k: (Z - mean) @ pca[:k].T  # noqa: E731
    psi, scale, nfeat = make(chart, X, Y, q)
    Zc, worst = test[:, 0].copy(), []
    for n in range(1, STEPS):
        Zc = np.clip(psi(Zc), -40, 5)
        worst.append(float(np.abs(Zc - test[:, n]).max()))
    run = np.maximum.accumulate(worst)
    rho = max(np.abs(np.linalg.eigvals(jacobian(psi, z))).max() for z in Xe[rng.choice(len(Xe), 60, replace=False)])
    rows.append({"k": k, "degree": q, "features": nfeat, "one_step": float(np.abs(psi(Xe) - Ye).max()), "rollout": {str(m): float(run[m - 1]) for m in (1, 10, 40, STEPS - 1)}, "rho": float(rho)})
    r = rows[-1]
    print(f"PCA-{k}    {q:3d} {nfeat:4d} {r['one_step']:9.2e} | " + " ".join(f"{v:9.2e}" for v in r["rollout"].values()) + f" | {rho:8.3f}", flush=True)
out["cases"] = rows
(HERE / "a2_hires_followup.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
