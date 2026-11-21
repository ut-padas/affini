"""
Theorem 2 (memo section T2): how a surrogate's one-step defect delta = Psi - Phi accumulates when the surrogate
is chained, on maps known exactly, with synthetic defects of chosen direction and shape.

(a) Davis-Skodje, x' = -x, y' = -g y + ((g - 1) x + g x^2) / (1 + x)^2 (g = 10, tau = 0.1): exact flow
    x(t) = x0 e^{-t}, y(t) = (y0 - x0/(1 + x0)) e^{-g t} + x(t)/(1 + x(t)); a stable fixed point at 0 with a slow
    eigenvalue e^{-tau} (eigenvector (1, 1), the slow manifold's tangent) and a fast one e^{-g tau} (eigenvector
    (0, 1)). Prediction (Proposition 2.2): the rollout error tends to (I - DPhi(0))^{-1} delta for a constant
    defect, up to O(eps^2): a slow defect is amplified by 1/(1 - e^{-tau}) ~ 10.5, a fast one by ~ 1.6; the
    classical worst case eps * A_n (A_n = sum of Jacobian-product norms) is reported beside it.
(b) A neutral map, the rotation of the harmonic oscillator by tau per step, with a defect delta(y) = eps B y
    (a linear field) or eps (sin(k phi), cos(k phi)) of the phase phi. Prediction (Proposition 2.3): the rollout
    error grows like n |<U^{-j} delta(U^j y0)>| (the defect's average in the rotating frame) and stays bounded
    when that average is zero; the classical bound grows like n eps for both.
"""

import numpy as np
from common import write

G, TAU = 10.0, 0.1


def ds_flow(p: np.ndarray, t: float = TAU) -> np.ndarray:
    x0, y0 = p
    x = x0 * np.exp(-t)
    return np.array([x, (y0 - x0 / (1 + x0)) * np.exp(-G * t) + x / (1 + x)])


def jac(f, p: np.ndarray, h: float = 1e-6) -> np.ndarray:
    return np.column_stack([(f(p + h * e) - f(p - h * e)) / (2 * h) for e in np.eye(len(p))])


def davis_skodje(eps: float = 1e-5, n: int = 400) -> dict:
    P = jac(ds_flow, np.zeros(2))
    lam = np.sort(np.linalg.eigvals(P).real)[::-1]
    out = {"eigenvalues": lam.tolist(), "cases": {}}
    for name, u in {"slow (1,1)/sqrt2": np.array([1.0, 1.0]) / np.sqrt(2), "fast (0,1)": np.array([0.0, 1.0]),
                    "x only (1,0)": np.array([1.0, 0.0])}.items():  # fmt: skip
        delta = eps * u
        x, y = np.array([1.0, 0.6]), np.array([1.0, 0.6])
        acc, Gprod, err = 0.0, [], []
        for _ in range(n):
            Jx = jac(ds_flow, x)
            Gprod = [Jx @ g for g in Gprod] + [np.eye(2)]
            acc = sum(np.linalg.norm(g, 2) for g in Gprod)
            x, y = ds_flow(x), ds_flow(y) + delta
            err.append(np.linalg.norm(y - x))
        pred = np.linalg.solve(np.eye(2) - P, delta)
        out["cases"][name] = {"rollout_final_over_eps": err[-1] / eps, "predicted_over_eps": float(np.linalg.norm(pred)) / eps,
                              "worst_case_A_n": float(acc), "relative_gap": float(abs(err[-1] - np.linalg.norm(pred)) / np.linalg.norm(pred))}  # fmt: skip
    return out


def rotation(eps: float = 1e-6, n: int = 4000, tau: float = 0.1) -> dict:
    c, s = np.cos(tau), np.sin(tau)
    U = np.array([[c, -s], [s, c]])
    y0 = np.array([1.0, 0.0])
    defects = {
        "linear, B = [[0.3, 0.2], [-0.1, 0.5]]": (lambda y: np.array([[0.3, 0.2], [-0.1, 0.5]]) @ y),
        "trace-free symmetric B = [[1, 0], [0, -1]]": (lambda y: np.array([[1.0, 0.0], [0.0, -1.0]]) @ y),
        "phase harmonic k = 3": (lambda y: np.array([np.sin(3 * np.arctan2(y[1], y[0])), np.cos(3 * np.arctan2(y[1], y[0]))])),
        "constant (1, 0)": (lambda y: np.array([1.0, 0.0])),
    }
    out = {"tau": tau, "n": n, "cases": {}}
    for name, g in defects.items():
        x, y = y0.copy(), y0.copy()
        errs, pulled = [], np.zeros(2)
        for j in range(n):
            pulled += np.linalg.matrix_power(U.T, j) @ g(np.linalg.matrix_power(U, j) @ y0)
            x, y = U @ x, U @ y + eps * g(y)
            errs.append(np.linalg.norm(y - x))
        mean = np.linalg.norm(pulled / n)
        errs = np.array(errs) / eps
        out["cases"][name] = {"orbit_mean_norm": float(mean), "rollout_over_eps_at_n": float(errs[-1]), "predicted_n_times_mean": float(n * mean),
                              "max_rollout_over_eps": float(errs.max()), "classical_bound_n": n}  # fmt: skip
    return out


def main() -> None:
    out = {"davis_skodje": davis_skodje(), "rotation": rotation()}
    path = write("t2_rollout", out)
    print("Davis-Skodje eigenvalues", np.round(out["davis_skodje"]["eigenvalues"], 4))
    for k, v in out["davis_skodje"]["cases"].items():
        print(f"  {k:18s} rollout/eps={v['rollout_final_over_eps']:.3f} predicted={v['predicted_over_eps']:.3f} "
              f"A_n={v['worst_case_A_n']:.2f} gap={v['relative_gap']:.1e}")  # fmt: skip
    for k, v in out["rotation"]["cases"].items():
        print(f"  {k:42s} mean={v['orbit_mean_norm']:.4f} rollout/eps(n)={v['rollout_over_eps_at_n']:.2f} "
              f"n*mean={v['predicted_n_times_mean']:.2f} max={v['max_rollout_over_eps']:.2f} classical={v['classical_bound_n']}")  # fmt: skip
    print("wrote", path)


if __name__ == "__main__":
    main()
