"""
Test 2 (HIRES, k = 3): the reduced surrogates on states kicked off the visited set, as an operator-split simulation does.
For 400 held-out visited states z and a kick delta of max-norm eps (random signs in every log abundance; or the same kick restricted to
the chart's span V, or to its orthogonal complement), compare with the solver at the kicked state:
  retained   max |Phi(z + delta) - Phi(z)|        what the exact step keeps of the kick (0 if the visited set attracted in one step);
  error      max |Psi(z + delta) - Phi(z + delta)| for the reduced ChebTT of Demonstration 8.5, the ideal reduced map Phi(D(pi .)) and a
             data-only degree-6 fit; and the share of kicked states each refuses.
"""
import json

import numpy as np
from common import HERE, FnModel, load_reduced, poly, setup

from src.benchmark.step_maps import Projected

s = setup()
p, test, mu, V, lo, hi, W = s["p"], s["test"], s["mu"], s["V"], s["lo"], s["hi"], s["W"]
Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
idx = np.random.default_rng(0).choice(len(Xe), 400, replace=False)
Z, Zn = Xe[idx], Ye[idx]
c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
coef6 = np.linalg.lstsq(poly((s["U"] - c) / h, 6), s["Y"], rcond=None)[0]
models = {"reduced ChebTT (8.5)": load_reduced(s),
          "ideal Phi(D(pi z))": Projected(FnModel(lambda U: p.step(poly(U, s["q"]) @ s["coef"]), mu, V, lo, hi), p, W),
          "data-only degree 6": Projected(FnModel(lambda U: poly((U - c) / h, 6) @ coef6, mu, V, lo, hi), p, W)}
rng = np.random.default_rng(1)
P = V @ V.T
rows = []
for kind in ("all components", "in the chart's span", "normal to the chart"):
    for eps in (0.0, 1e-3, 1e-2, 5e-2):
        d = rng.choice([-1.0, 1.0], Z.shape)
        d = d if kind == "all components" else d @ P if kind == "in the chart's span" else d - d @ P
        d = eps * d / np.maximum(np.abs(d).max(axis=1, keepdims=True), 1e-300)
        Zk = Z + d
        true = p.step(Zk)
        m = p.scored(true)
        row = {"kick": kind, "eps": eps, "retained": float(np.max(np.where(m, np.abs(true - Zn), 0.0)))}
        for name, model in models.items():
            r = np.asarray(model.refuses(Zk), dtype=bool)
            e = np.max(np.where(m, np.abs(Zk + model.predict(Zk) - true), 0.0), axis=1)
            row[name] = {"worst": float(e.max()), "median": float(np.median(e)), "worst_answered": float(e[~r].max()) if (~r).any() else None,
                         "refused_percent": float(100 * r.mean())}
        rows.append(row)
        print(f"{kind:22s} eps {eps:7.0e}: retained {row['retained']:.2e} | " + " | ".join(
            f"{n.split()[0]} worst {row[n]['worst']:.2e} med {row[n]['median']:.2e} ref {row[n]['refused_percent']:.0f}%" for n in models), flush=True)
(HERE / "t2_offset.json").write_text(json.dumps(rows, indent=1))
