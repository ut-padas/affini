"""Where does max ||D Psi|| = 1.27 at every degree of the "gradients" fit come from?"""

import json

import numpy as np
from common import P, Projected, Recipe, fit, jac_of, setup, spec

s = setup()
Zs = s["X"][s["js"][:200]]
Jd = s["Jd"][:200]
out = {}


class Frozen:
    """A model whose next state does not depend on the input: what is left of D Psi is the invariant projection's dependence on W y."""

    def __init__(self, nxt):
        self.nxt = nxt

    def refuses(self, Z):
        return np.zeros(len(Z), dtype=bool)

    def predict(self, Z):
        return self.nxt - Z


for name in ("data", "gradients"):
    for q in (4, 6, 8):
        G = fit(s, name, q)
        sur = s["mk"](G)
        raw = sur.model
        D = jac_of(sur, Zs)
        n = spec(D)
        i = int(np.argmax(n))
        Draw = jac_of(raw, Zs)  # without the invariant projection
        nxt = Zs + raw.predict(Zs)
        Dproj = jac_of(Projected(Frozen(nxt), P, s["W"]), Zs)  # the projection's part alone
        u = (Zs - s["mu"]) @ s["V"]
        clipped = bool(np.any((u <= s["lo"]) | (u >= s["hi"])))
        row = {"max": float(n[i]), "argmax_index": i, "argmax_time_step": int(s["js"][i] // s["train"].shape[0]),
               "no_projection_max": float(spec(Draw).max()), "no_projection_at_argmax": float(spec(Draw)[i]),
               "projection_only_max": float(spec(Dproj).max()), "projection_only_at_argmax": float(spec(Dproj)[i]),
               "dynamics_at_argmax": float(spec(Jd)[i]), "median": float(np.median(n)), "dynamics_median": float(np.median(spec(Jd))),
               "y7_y8_in_at_argmax": (10.0 ** Zs[i, 6:]).tolist(), "y7_y8_out_at_argmax": (10.0 ** nxt[i, 6:]).tolist(),
               "any_chart_clipped": clipped,
               "worst_ratio_to_dynamics_same_state": float(np.max(n / spec(Jd))),
               "max_entry_error_vs_dynamics": float(np.max(np.abs(D - Jd)))}  # fmt: skip
        if name == "gradients":
            corr = s["mk"](G, s["Bcoef"])
            Dc = jac_of(corr, Zs)
            row["corrected_max"] = float(spec(Dc).max())
            row["corrected_max_entry_error_vs_dynamics"] = float(np.max(np.abs(Dc - Jd)))
            row["corrected_worst_ratio_to_dynamics_same_state"] = float(np.max(spec(Dc) / spec(Jd)))
        out[f"{name}_q{q}"] = row
        print(name, q, json.dumps(row), flush=True)
out["dynamics_max"] = float(spec(Jd).max())
out["dynamics_argmax_time_step"] = int(s["js"][int(np.argmax(spec(Jd)))] // s["train"].shape[0])
print(out["dynamics_max"], out["dynamics_argmax_time_step"])
json.dump(out, open("t1_dpsi.json", "w"), indent=1)
