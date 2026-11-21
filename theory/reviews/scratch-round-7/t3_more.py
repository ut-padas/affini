"""(a) the data fit on all test trajectories, like for like; (b) degrees above the selection's range; (c) solve timings on the same states."""

import json
import time

import numpy as np
from common import P, fd_jacobian, fit, log_jacobian, rollout, setup

from src.benchmark.step_maps import ROLLOUTS

out = {}
s = setup(3, 300, "sens")
train, test = s["train"], s["test"]
for q in (5, 6):
    sur = s["mk"](fit(s, "data", q))
    out[f"data_q{q}"] = {"test_20": rollout(P, sur, test[:ROLLOUTS])["rollout_worst"], "test_last_20": rollout(P, sur, test[ROLLOUTS:])["rollout_worst"],
                         "train_all": rollout(P, sur, train)["rollout_worst"]}  # fmt: skip
    print("data", q, out[f"data_q{q}"], flush=True)
for seed, n, jac in ((3, 300, "sens"), (11, 300, "sens"), (12, 300, "sens")):
    s = setup(seed, n, jac)
    for q in (8, 9, 10, 11, 12):
        G = fit(s, "gradients", q)
        corr = s["mk"](G, s["Bcoef"])
        per = [rollout(P, corr, test[i : i + 1])["rollout_worst"] for i in range(len(test))]
        row = {"terms": len(G.idx), "train_chain_corrected_all": rollout(P, corr, train)["rollout_worst"], "test_20": max(per[:ROLLOUTS]),
               "test_all": max(per), "test_trajectories_over_tol": int(np.sum(np.array(per) > 0.01)), "per_trajectory": per}  # fmt: skip
        out[f"gradients_seed{seed}_q{q}"] = row
        print(seed, q, {k: v for k, v in row.items() if k != "per_trajectory"}, flush=True)
# timings on the same 100 training states, one at a time
Z = s["X"][s["js"][:100]]
t = {}
for name, f in (("plain", lambda z: P.step(z[None])), ("sensitivity_8", log_jacobian), ("central_differences_16", fd_jacobian)):
    ts = []
    for z in Z:
        t0 = time.perf_counter()
        f(z)
        ts.append(time.perf_counter() - t0)
    t[name] = {"median_ms": 1e3 * float(np.median(ts)), "mean_ms": 1e3 * float(np.mean(ts))}
out["timings"] = t
print(t)
json.dump(out, open("t3_more.json", "w"), indent=1)
