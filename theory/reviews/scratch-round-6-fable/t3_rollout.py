"""
Test 3 (HIRES, k = 3): is the rollout error of Demonstration 8.5's reduced ChebTT the exact dynamics' amplification of its one-step
defects (Proposition 2.2's first-order sum), or the surrogate's own instability?
For the benchmark's 20 rollout trajectories: the rollout error by step; for the TRAJ worst of them, the step map's Jacobians J_j along
the exact chain (central differences, as `dynamics`) and
  first order (exact J)      e_n = sum_j J_{n-1} ... J_{j+1} delta_j, delta_j = Psi(x_j) - x_{j+1} (the defect on the exact chain);
  first order (surrogate J)  the same with DPsi(x_j) in place of J_j (the surrogate's own propagation of the same defects);
and the split of the measured error into its chart part |V V^T e| and normal part. The same for a data-only degree-6 fit (stable) and
degree-8 fit (diverges), as controls.
"""
import json

import numpy as np
from common import HERE, FnModel, load_reduced, poly, setup

from src.benchmark.step_maps import ROLLOUTS, Projected

TRAJ = 6
s = setup()
p, test, mu, V, lo, hi, W = s["p"], s["test"], s["mu"], s["V"], s["lo"], s["hi"], s["W"]
true = test[:ROLLOUTS]
n = true.shape[1] - 1
c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
models = {"reduced ChebTT (8.5)": load_reduced(s)}
for q in (6, 8):
    cf = np.linalg.lstsq(poly((s["U"] - c) / h, q), s["Y"], rcond=None)[0]
    models[f"data-only degree {q}"] = Projected(FnModel(lambda U, cf=cf, q=q: poly((U - c) / h, q) @ cf, mu, V, lo, hi), p, W)


def chain(model):
    Z, out = true[:, 0].copy(), []
    for k in range(n):
        Z = Z + model.predict(Z)
        out.append(Z - true[:, k + 1])
    return np.array(out)  # (n, traj, d)


def jac(f, z, hh=1e-5):
    E = np.zeros((16, 8))
    for i in range(8):
        E[2 * i, i], E[2 * i + 1, i] = hh, -hh
    o = f(z[None] + E)
    return (o[0::2] - o[1::2]).T / (2 * hh)


errs = {name: chain(m) for name, m in models.items()}
main = errs["reduced ChebTT (8.5)"]
worst = np.argsort(-np.abs(main).max(axis=(0, 2)))[:TRAJ]
out = {"per_trajectory_worst": {name: np.abs(e).max(axis=(0, 2)).tolist() for name, e in errs.items()}, "trajectories": {}}
P = V @ V.T
for t in worst:
    Js = [jac(p.step, true[t, k]) for k in range(n)]
    row = {}
    for name, m in models.items():
        delta = np.array([(true[t, k] + m.predict(true[t, k][None])[0]) - true[t, k + 1] for k in range(n)])
        Jm = [jac(lambda Zq, m=m: Zq + m.predict(Zq), true[t, k], 1e-4) for k in range(n)]
        fo = {}
        for label, JJ in (("exact_J", Js), ("surrogate_J", Jm)):
            e, worst_e = np.zeros(8), 0.0
            for k in range(n):
                e = JJ[k] @ e + delta[k]
                worst_e = max(worst_e, float(np.abs(e).max()))
            fo[label] = worst_e
        e = errs[name][:, t]
        k_star = int(np.abs(e).max(axis=1).argmax())
        row[name] = {"measured_worst": float(np.abs(e).max()), "at_step": k_star + 1, "defect_worst": float(np.abs(delta).max()),
                     "first_order_exact_J": fo["exact_J"], "first_order_surrogate_J": fo["surrogate_J"],
                     "chart_part_at_worst": float(np.abs(e[k_star] @ P).max()), "normal_part_at_worst": float(np.abs(e[k_star] - e[k_star] @ P).max()),
                     "max_norm_DPsi": float(max(np.linalg.norm(J, 2) for J in Jm)), "max_norm_DPhi": float(max(np.linalg.norm(J, 2) for J in Js))}
        print(int(t), name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row[name].items()}, flush=True)
    out["trajectories"][int(t)] = row
(HERE / "t3_rollout.json").write_text(json.dumps(out, indent=1))
