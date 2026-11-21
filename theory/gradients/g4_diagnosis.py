"""
The diagnosis behind g4_hires_recipe.py's fits "consistent" and "gradients": at 40 kicked test states, the first-order normal correction
with the true Phi(D(u)) against the degree-6 data fit G(u), and the forward-sensitivity Jacobian against central differences of the step.
"""

import json

import numpy as np
from g2_hires import OUT, ChartPoly, P, poly
from g4_hires_recipe import log_jacobian

from src.benchmark.step_maps import data, visited


def main() -> None:
    train, test = data("hires")
    X = visited(train)
    Y = train[:, 1:].transpose(1, 0, 2).reshape(-1, X.shape[1])
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:3].T
    U = (X - mu) @ V
    dec = np.linalg.lstsq(poly(U, 4), X, rcond=None)[0]

    def decode(W):
        return poly(W, 4) @ dec

    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    G = ChartPoly(6, lo - pad, hi + pad).fit(U, Y)
    Xe = test[:, :-1].reshape(-1, X.shape[1])
    rng = np.random.default_rng(5)
    idx = rng.choice(len(Xe), 40, replace=False)
    out = {"training_singular_values": np.linalg.svd(X - mu, compute_uv=False).tolist(), "kicks": []}
    for eps in (0.003, 0.01):
        Zk = Xe[idx] + eps * rng.choice([-1.0, 1.0], (40, X.shape[1]))
        Yk = P.step(Zk)
        Uk = (Zk - mu) @ V
        n = Zk - decode(Uk)
        PhiD = P.step(decode(Uk))
        Jx = np.array([log_jacobian(z) for z in decode(Uk)])
        Jn = np.einsum("nij,nj->ni", Jx, n)
        sc = P.scored(Yk)
        err = {k: float(np.max(np.where(sc, np.abs(E - Yk), 0.0))) for k, E in
               (("data_fit", G(Uk)), ("phi_of_decoded", PhiD), ("phi_of_decoded_plus_Jn", PhiD + Jn), ("data_fit_plus_Jn", G(Uk) + Jn))}  # fmt: skip
        z, h = decode(Uk[:1])[0], 1e-5
        J = log_jacobian(z)
        Jfd = np.stack([(P.step((z + h * e)[None])[0] - P.step((z - h * e)[None])[0]) / (2 * h) for e in np.eye(X.shape[1])], axis=1)
        out["kicks"].append({"eps": eps, "errors": err, "normal_max": float(np.abs(n).max()), "Jn_max": float(np.abs(Jn).max()),
                             "jacobian_vs_central_differences": float(np.abs(J - Jfd).max()),
                             "jacobian_eigenvalue_moduli": np.sort(np.abs(np.linalg.eigvals(J)))[::-1].tolist()})  # fmt: skip
        print(f"eps {eps}: " + ", ".join(f"{k} {v:.2e}" for k, v in err.items()) + f"; |n| {np.abs(n).max():.2e}; Jacobian vs differences {out['kicks'][-1]['jacobian_vs_central_differences']:.1e}")
    (OUT / "g4_diagnosis.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
