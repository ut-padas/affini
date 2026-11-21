"""
Round-3 test A: Theorem 8's practical claim on HIRES (8 species, one linear invariant y7 + y8 that varies between
trajectories, a two-parameter family of initial states: several invariant leaves, a visited set of dimension 3).

The benchmark's setting: log10 abundances, dt = 2, burn 5, 170 steps, 100 training and 40 held-out trajectories.
Surrogates, all least-squares polynomials fitted on the training trajectories' visited states only:
  slow   Psi = G(W^T (z - mean)): total degree q in k learned linear coordinates, next state directly. Charts W:
           PCA  the k leading principal directions of the visited states,
           JAC  the k leading eigenvectors of the mean of DPhi^T DPhi over 150 visited states (the dominant right
                singular subspace of the step map's Jacobian, central differences),
           INV  log10(y7 + y8) (the invariant, a known function of the state) and the k - 1 leading JAC directions.
  full   total degree 2 or 3 in the 8 coordinates, next state directly or as an increment z + g(z).
Measured on the held-out trajectories: the worst one-step error on visited states, on states displaced by up to 0.05 dex
in every coordinate (the tube), the worst rollout error after 1, 10, 40 and 169 steps, the drift of the invariant along
the chain, and the spectral radius and norm of DPsi at visited states (the surrogate's gain), against DPhi's.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from src.problems.ode.chemistry import hires_initial, hires_jac, hires_rhs  # noqa: E402

DT, BURN, STEPS, R = 2.0, 5.0, 170, 0.05
HERE = Path(__file__).resolve().parent


def flow(y, t_eval):
    s = solve_ivp(lambda _t, v: hires_rhs(v), (0, t_eval[-1]), y, method="Radau", jac=lambda _t, v: hires_jac(v), t_eval=t_eval, rtol=1e-10, atol=1e-16)
    return np.log10(np.maximum(s.y.T, 1e-30))


def trajectories(Y0):
    t = BURN + DT * np.arange(STEPS + 1)
    return np.array([flow(y, np.concatenate([[0.0], t]))[1:] if BURN > 0 else flow(y, t) for y in Y0])


def step(Z):
    return np.array([flow(10.0**z, np.array([DT]))[-1] for z in Z])


def monomials(k, q):
    return [a for a in np.ndindex(*([q + 1] * k)) if sum(a) <= q]


def features(U, q, idx):
    V = np.polynomial.chebyshev.chebvander(np.clip(U, -1.5, 1.5), q)
    return np.stack([np.prod([V[:, c, a[c]] for c in range(U.shape[1])], axis=0) for a in idx], axis=1)


def make(chart, X, Y, q, increment=False):
    """Least-squares Psi = G(chart(z)) (+ z if increment), features scaled to the visited range of the chart's coordinates."""
    C = chart(X)
    lo, hi = C.min(0), C.max(0)
    idx = monomials(C.shape[1], q)
    scale = lambda Z: 2 * (chart(Z) - lo) / (hi - lo) - 1  # noqa: E731
    coef = np.linalg.lstsq(features(scale(X), q, idx), Y - X if increment else Y, rcond=1e-12)[0]
    psi = lambda Z: (Z if increment else 0.0) + features(scale(Z), q, idx) @ coef  # noqa: E731
    return psi, scale, len(idx)


def jacobian(f, z, h=1e-4):
    E = np.eye(len(z)) * h
    return (f(z + E) - f(z - E)).T / (2 * h)


def main():
    t0 = time.time()
    train, test = trajectories(hires_initial(100, 1)), trajectories(hires_initial(40, 2))
    X, Y = train[:, :-1].reshape(-1, 8), train[:, 1:].reshape(-1, 8)
    Xe, Ye = test[:, :-1].reshape(-1, 8), test[:, 1:].reshape(-1, 8)
    print(f"trajectories {time.time() - t0:.0f}s; visited box (dex):", np.round(X.max(0) - X.min(0), 2), flush=True)
    rng = np.random.default_rng(0)
    mean = X.mean(0)
    sv = np.linalg.svd(X - mean, compute_uv=False)
    pca = np.linalg.svd(X - mean, full_matrices=False)[2]
    print("PCA singular values / first:", np.round(sv / sv[0], 5))
    pick = rng.choice(len(X), 150, replace=False)
    D = np.array([jacobian(step, z) for z in X[pick]])
    sig = np.array([np.linalg.svd(d, compute_uv=False) for d in D])
    print(f"DPhi {time.time() - t0:.0f}s; singular values of DPhi (median over 150 visited states):", np.array2string(np.median(sig, 0), precision=3))
    print("   max over states:", np.array2string(sig.max(0), precision=3), " spectral radius max", f"{max(np.abs(np.linalg.eigvals(d)).max() for d in D):.3f}")
    w, Vj = np.linalg.eigh(np.mean([d.T @ d for d in D], axis=0))
    jac = Vj[:, ::-1].T
    print("eigenvalues of mean DPhi^T DPhi (sqrt):", np.array2string(np.sqrt(np.maximum(w[::-1], 0)), precision=3))
    # how many coordinates the visited set needs: tangent rank of the visited set (local PCA) is reported through the fits below

    inv = lambda Z: np.log10(10.0 ** Z[:, 6] + 10.0 ** Z[:, 7])  # noqa: E731
    charts = {}
    for k in (2, 3, 4, 5):
        charts[f"PCA-{k}"] = lambda Z, k=k: (Z - mean) @ pca[:k].T
        charts[f"JAC-{k}"] = lambda Z, k=k: (Z - mean) @ jac[:k].T
        charts[f"INV-{k}"] = lambda Z, k=k: np.column_stack([inv(Z), (Z - mean) @ jac[: k - 1].T])
    degree = {2: 8, 3: 6, 4: 5, 5: 4}
    full = lambda Z: Z  # noqa: E731
    cases = [(name, c, degree[int(name[-1])], False) for name, c in charts.items()]
    cases += [("full-2", full, 2, False), ("full-2-incr", full, 2, True), ("full-3", full, 3, False), ("full-3-incr", full, 3, True)]

    tube_pick = rng.choice(len(Xe), 300, replace=False)
    Xt = Xe[tube_pick] + rng.uniform(-R, R, (300, 8))
    Yt = step(Xt)
    gain_pick = rng.choice(len(Xe), 100, replace=False)
    Dphi = [jacobian(step, z) for z in Xe[gain_pick[:30]]]
    print(f"exact map: spectral radius {max(np.abs(np.linalg.eigvals(d)).max() for d in Dphi):.3f}, norm {max(np.linalg.norm(d, 2) for d in Dphi):.3f}", flush=True)
    out = {"dphi_singular_median": np.median(sig, 0).tolist(), "pca_singular": (sv / sv[0]).tolist(), "cases": []}
    marks = (1, 10, 40, STEPS - 1)
    print(f"{'surrogate':12s} {'feat':>4s} {'1-step':>9s} {'tube':>9s} | rollout at " + " ".join(f"{m:>9d}" for m in marks) + f" | {'inv drift':>9s} {'rho(DPsi)':>9s} {'|DPsi|':>8s} {'out':>5s}")
    for name, chart, q, inc in cases:
        psi, scale, nfeat = make(chart, X, Y, q, inc)
        Zc, worst, drift, outside = test[:, 0].copy(), [], 0.0, 0.0
        for k in range(1, STEPS):
            outside = max(outside, float(np.mean(np.abs(scale(Zc)).max(1) > 1.0)))
            Zc = np.clip(psi(Zc), -40, 5)
            worst.append(float(np.abs(Zc - test[:, k]).max()))
            drift = max(drift, float(np.abs(inv(Zc) - inv(test[:, k])).max()))
        run = np.maximum.accumulate(worst)
        Dpsi = [jacobian(psi, z) for z in Xe[gain_pick]]
        row = {"name": name, "features": nfeat, "one_step_visited": float(np.abs(psi(Xe) - Ye).max()), "one_step_tube": float(np.abs(psi(Xt) - Yt).max()),
               "rollout": {str(m): float(run[m - 1]) for m in marks}, "invariant_drift": drift,
               "spectral_radius": float(max(np.abs(np.linalg.eigvals(d)).max() for d in Dpsi)), "norm": float(max(np.linalg.norm(d, 2) for d in Dpsi)),
               "outside_fraction": outside}  # fmt: skip
        out["cases"].append(row)
        print(f"{name:12s} {nfeat:4d} {row['one_step_visited']:9.2e} {row['one_step_tube']:9.2e} |            " + " ".join(f"{row['rollout'][str(m)]:9.2e}" for m in marks)
              + f" | {drift:9.2e} {row['spectral_radius']:9.3f} {row['norm']:8.2f} {outside:5.2f}", flush=True)  # fmt: skip
    (HERE / "a_hires_chart.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
