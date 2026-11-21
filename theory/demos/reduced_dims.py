"""
How many chart coordinates a reduced surrogate of a stiff step map needs (memo T6 and section 8), measured before any surrogate is
built, on the benchmark's own trajectories. Usage: reduced_dims.py <problem> [<problem> ...] [--tol T].

For the principal directions of the training trajectories' visited states (mean mu, directions V) and k = 1, 2, ...:
  lift error      max |Phi(mu + V_k V_k^T (z - mu)) - Phi(z)| over 400 held-out visited states (one solver step each): the error of
                  a surrogate whose input is the k chart coordinates and whose output is a full state (the fast part of the dropped
                  directions is damped by the step; Proposition 8.2's middle term);
  residual        max |z - mu - V_k V_k^T (z - mu)| over the held-out visited states (no solve): the error of decoding a predicted chart
                  value back to a state, which a surrogate with k outputs (latent dynamics) also pays.
Prediction (Theorem 6.6, review 1 test E): for the Fisher-KPP ladder the k at which the lift error reaches the tolerance does not
grow with the grid (d = 10 against d = 20); for HIRES and POLLU it is the dimension of the visited set (review 3 test A: 3 for HIRES).
Errors are over the problem's scored entries, in its coordinates (dex for log systems).
"""

import json
import sys
from pathlib import Path

import numpy as np

from src.benchmark.step_maps import STEP_MAPS, data, visited

OUT = Path(__file__).resolve().parent / "data"


def first_k(rows: list, key: str, t: float):
    return next((r["k"] for r in rows if r[key] <= t), None)


def main() -> None:
    args = sys.argv[1:]
    tol = float(args[args.index("--tol") + 1]) if "--tol" in args else 0.01
    names = [a for a in args if not a.startswith("--") and a != str(tol)]
    OUT.mkdir(exist_ok=True)
    for name in names:
        p = STEP_MAPS[name]
        train, test = data(name)
        X = visited(train)
        Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
        mu = X.mean(axis=0)
        _, sv, Vt = np.linalg.svd(X - mu, full_matrices=False)
        idx = np.random.default_rng(0).choice(len(Xe), min(400, len(Xe)), replace=False)
        m_in, m_out = p.scored(Xe[idx]), p.scored(Ye[idx])
        rows = []
        for k in range(1, X.shape[1] + 1):
            P = Vt[:k].T @ Vt[:k]
            proj = mu + (Xe[idx] - mu) @ P
            lift = float(np.max(np.where(m_out, np.abs(p.step(proj) - Ye[idx]), 0.0)))
            resid = float(np.max(np.where(m_in, np.abs(proj - Xe[idx]), 0.0)))
            rows.append({"k": k, "lift_error": lift, "residual": resid})
            if lift <= tol / 10 and resid <= tol / 10:
                break
        out = {"problem": name, "d": int(X.shape[1]), "tol": tol, "singular_values": (sv / sv[0]).tolist(), "rows": rows,
               "k_lift_quarter_tol": first_k(rows, "lift_error", tol / 4), "k_residual_quarter_tol": first_k(rows, "residual", tol / 4)}  # fmt: skip
        (OUT / f"reduced_dims_{name}.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(f"{name} (d = {out['d']}): k for lift error <= tol/4: {out['k_lift_quarter_tol']}; for residual <= tol/4: {out['k_residual_quarter_tol']}")
        print("   " + "  ".join(f"k{r['k']}: {r['lift_error']:.1e}/{r['residual']:.1e}" for r in rows), flush=True)


if __name__ == "__main__":
    main()
