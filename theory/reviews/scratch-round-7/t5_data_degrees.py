"""The value-only data fit (no solve) at degrees 6-12: rollouts on the benchmark's 20 and on all 40 test trajectories, the training chains,
the one-step error and max ||D Psi|| at 200 training states."""

import json

import numpy as np
from common import P, fit, jac_of, rollout, setup, spec

from src.benchmark.step_maps import ROLLOUTS

s = setup(3, 300, "sens")
train, test = s["train"], s["test"]
Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
Zs = s["X"][s["js"][:200]]
out = {}
for q in range(6, 13):
    G = fit(s, "data", q)
    sur = s["mk"](G)
    per = [rollout(P, sur, test[i : i + 1])["rollout_worst"] for i in range(len(test))]
    out[q] = {"terms": len(G.idx), "one_step": float(np.max(np.where(P.scored(Ye), np.abs(Xe + sur.predict(Xe) - Ye), 0.0))),
              "train_chain_40": rollout(P, sur, train[:40])["rollout_worst"], "train_chain_all": rollout(P, sur, train)["rollout_worst"],
              "test_20": max(per[:ROLLOUTS]), "test_all": max(per), "over_tol": int(np.sum(np.array(per) > 0.01)),
              "dpsi": float(spec(jac_of(sur, Zs)).max())}  # fmt: skip
    print(q, out[q], flush=True)
json.dump(out, open("t5_data_degrees.json", "w"), indent=1)
