"""Degrees 8-12 of the "gradients" fit and of the value-only fits (seed 3, as committed): condition (ii) for the corrected surrogate, the
plain rollout, and the kicked errors on g4_hires_recipe.py's own kicked states."""

import json

import numpy as np
from common import P, fit, jac_of, rollout, setup, spec

EPS = (0.003, 0.01, 0.03, 0.05)
s = setup(3, 300, "sens")
train, test, rng = s["train"], s["test"], s["rng"]
Xe = test[:, :-1].reshape(-1, 8)
kick_idx = rng.choice(len(Xe), 300, replace=False)
signs = {eps: rng.choice([-1.0, 1.0], Xe[kick_idx].shape) for eps in EPS}
Zs = s["X"][s["js"][:200]]
out = {}
for name in ("gradients", "consistent", "data"):
    for q in (8, 9, 10, 11, 12):
        G = fit(s, name, q)
        plain, corr = s["mk"](G), s["mk"](G, s["Bcoef"])
        row = {"dpsi_plain": float(spec(jac_of(plain, Zs)).max()), "dpsi_corrected": float(spec(jac_of(corr, Zs)).max()),
               "test_all_plain": rollout(P, plain, test)["rollout_worst"], "test_all_corrected": rollout(P, corr, test)["rollout_worst"],
               "train_all_plain": rollout(P, plain, train)["rollout_worst"], "kicks": {}}  # fmt: skip
        for eps in EPS:
            Zk = Xe[kick_idx] + eps * signs[eps]
            Yk = P.step(Zk)
            row["kicks"][eps] = float(np.max(np.where(P.scored(Yk), np.abs(Zk + corr.predict(Zk) - Yk), 0.0)))
        row["slope_corrected"] = float(np.polyfit(np.log(EPS), np.log([row["kicks"][e] for e in EPS]), 1)[0])
        out[f"{name}_q{q}"] = row
        print(name, q, json.dumps(row), flush=True)
        if name != "gradients" and q >= 9:
            break
json.dump(out, open("t4_high_degree.json", "w"), indent=1)
