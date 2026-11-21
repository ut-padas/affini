"""
The gradient tier on a stiff step map (memo section G): a reduced surrogate of HIRES fitted with values only against values and
Jacobians, in the 3 chart coordinates of Demonstration 8.5.

Construction: chart pi(z) = V^T (z - mu) (top-3 principal directions of the visited log-states); lift D(u) (a least-squares polynomial of
total degree 4 from chart values to log-states); oracle u -> Phi(D(u)) in log coordinates; its Jacobian in u is
DPhi_log(D(u)) DD(u), with DPhi_log = diag(1/y') M diag(y), M the forward sensitivity of the physical step (the state and its 8 x 8
sensitivity integrated together by Radau), and DD(u) the decoder's derivative. The surrogate is a total-degree-q Chebyshev polynomial
in the chart coordinates (scaled to the chart box), one per output, fitted by least squares on N chart values of training visited
states: values only (N equations per output), or values and the 3 chart derivatives (4N). It predicts the next state and is projected
onto the invariant y7 + y8 (the benchmark's Projected). Scored like the benchmark on the test trajectories: one-step worst, the error
surviving one step, the worst rollout of 20 trajectories over 170 steps. Cost: solver calls, and the measured wall time of a
sensitivity solve against a plain one.

Predictions, committed before the run (tolerance 0.01 dex):
  G2.1 at degree q = 8 the fit with values and Jacobians at N points matches the one-step accuracy of the value-only fit at about 4N
       points (1 + k equations per point, k = 3);
  G2.2 the gradient-enhanced fit holds a quarter of the tolerance (2.5e-3) in one step with at most a few hundred solves, over twenty
       times fewer than the value-only reduced ChebTT's 9,477 (Demonstration 8.5);
  G2.3 a sensitivity solve costs at most three plain solves in this Python implementation;
  G2.4 the rollout stays about 50-150 times the one-step error (the dynamics' amplification), so a smaller one-step error lowers the
       rollout in proportion, down to the lift's floor (about 2.5e-4).
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag

from src.benchmark.step_maps import (
    ROLLOUTS,
    STEP_MAPS,
    Projected,
    data,
    invariants_of,
    rollout,
    survived,
    visited,
)

OUT = Path(__file__).resolve().parent / "data"
P = STEP_MAPS["hires"]
S = P.system


def poly(U, q):
    cols = [np.ones(len(U))]
    for deg in range(1, q + 1):
        for idx in itertools.combinations_with_replacement(range(U.shape[1]), deg):
            cols.append(np.prod(U[:, list(idx)], axis=1))
    return np.stack(cols, axis=1)


def sensitivity_step(y0: np.ndarray, dt: float):
    """The physical step and its sensitivity M = dy(dt)/dy0, by Radau on the state and M together (a block-diagonal Newton matrix)."""
    d = len(y0)

    def f(_t, v):
        y, M = v[:d], v[d:].reshape(d, d)
        return np.concatenate([S.rhs(y), (S.jac(y) @ M).ravel()])

    def jac(_t, v):
        J = S.jac(v[:d])
        return block_diag(*([J] * (d + 1)))

    sol = solve_ivp(f, (0.0, dt), np.concatenate([y0, np.eye(d).ravel()]), method="Radau", jac=jac, rtol=S.rtol, atol=S.atol)
    return sol.y[:d, -1], sol.y[d:, -1].reshape(d, d)


class ChartPoly:
    """Per-output total-degree-q Chebyshev polynomials of the chart coordinates scaled to [-1, 1] on the chart box."""

    def __init__(self, q, lo, hi):
        self.q, self.lo, self.hi = q, lo, hi
        self.idx = [m for m in itertools.product(range(q + 1), repeat=len(lo)) if sum(m) <= q]

    def scale(self, U):
        return (2 * U - self.lo - self.hi) / (self.hi - self.lo)

    def design(self, U, grad: bool):
        s = self.scale(U)
        cheb, E = np.polynomial.chebyshev, np.eye(self.q + 1)
        T = np.stack([cheb.chebval(s, E[k]) for k in range(self.q + 1)], axis=-1)
        V = np.stack([np.prod([T[:, j, m[j]] for j in range(s.shape[1])], axis=0) for m in self.idx], axis=1)
        if not grad:
            return V, None
        dT = np.stack([cheb.chebval(s, cheb.chebder(E[k])) if k else np.zeros_like(s) for k in range(self.q + 1)], axis=-1)
        G = []
        for jj in range(s.shape[1]):
            factor = 2.0 / (self.hi[jj] - self.lo[jj])  # d s_jj / d u_jj
            G.append(factor * np.stack([np.prod([(dT if j == jj else T)[:, j, m[j]] for j in range(s.shape[1])], axis=0) for m in self.idx], axis=1))
        return V, G

    def fit(self, U, Y, dY=None):
        V, G = self.design(U, dY is not None)
        if dY is None:
            self.coef = np.linalg.lstsq(V, Y, rcond=None)[0]
            return self
        w = (self.hi - self.lo) / 2  # gradient rows in the units of a chart box half-width
        A = np.vstack([V] + [w[j] * G[j] for j in range(len(G))])
        B = np.vstack([Y] + [w[j] * dY[:, :, j] for j in range(len(G))])
        self.coef = np.linalg.lstsq(A, B, rcond=None)[0]
        return self

    def __call__(self, U):
        return self.design(U, False)[0] @ self.coef


class Reduced:
    """Psi(z) = G(pi(z)) as an increment model, refusing chart values outside the box."""

    def __init__(self, G, mu, V, lo, hi):
        self.G, self.mu, self.V, self.lo, self.hi, self.evals = G, mu, V, lo, hi, 0

    def refuses(self, Z):
        U = (Z - self.mu) @ self.V
        return np.any((self.lo > U) | (self.hi < U), axis=1)

    def predict(self, Z):
        return self.G(np.clip((Z - self.mu) @ self.V, self.lo, self.hi)) - Z


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train, test = data("hires")
    X = visited(train)
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:3].T
    U_all = (X - mu) @ V
    dec = np.linalg.lstsq(poly(U_all, 4), X, rcond=None)[0]
    decode = lambda U: poly(U, 4) @ dec  # noqa: E731
    lo, hi = U_all.min(0), U_all.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    W = invariants_of(P, train)
    Xe, Ye = test[:, :-1].reshape(-1, test.shape[2]), test[:, 1:].reshape(-1, test.shape[2])
    mask = P.scored(Ye)
    rng = np.random.default_rng(1)
    order = rng.permutation(len(U_all))
    Ns = [25, 50, 100, 200, 400, 800]
    Upool = U_all[order[: max(Ns)]]
    # one oracle pass over the pool: values and Jacobians (in u) of u -> Phi(D(u)), log coordinates
    Z0 = decode(Upool)
    h = 1e-6
    dD = np.stack([(decode(Upool + h * e) - decode(Upool - h * e)) / (2 * h) for e in np.eye(3)], axis=2)  # (n, 8, 3)
    vals, jacs, t_sens = [], [], []
    for z, dz in zip(Z0, dD, strict=True):
        y0 = 10.0**z
        t0 = time.perf_counter()
        y1, M = sensitivity_step(y0, P.dt)
        t_sens.append(time.perf_counter() - t0)
        vals.append(np.log10(np.maximum(y1, 1e-30)))
        jacs.append((M * (y0[None, :] / y1[:, None])) @ dz)
    vals, jacs = np.array(vals), np.array(jacs)
    t_plain = []
    for z in Z0[:50]:
        t0 = time.perf_counter()
        S.step(10.0 ** z[None, :], P.dt)
        t_plain.append(time.perf_counter() - t0)
    check = float(np.max(np.abs(vals[:50] - P.step(Z0[:50]))))  # the sensitivity solve's values against the benchmark's step
    out = {"k": 3, "decoder_degree": 4, "sensitivity_seconds_median": float(np.median(t_sens)), "plain_seconds_median": float(np.median(t_plain)),
           "sensitivity_over_plain": float(np.median(t_sens) / np.median(t_plain)), "value_check_against_step": check, "fits": []}  # fmt: skip
    for q in (6, 8):
        for N in Ns:
            for grad in (False, True):
                G = ChartPoly(q, lo, hi)
                if not grad and len(G.idx) > N:
                    continue
                G.fit(Upool[:N], vals[:N], jacs[:N] if grad else None)
                sur = Projected(Reduced(G, mu, V, lo, hi), P, W)
                e = np.max(np.where(mask, np.abs(Xe + sur.predict(Xe) - Ye), 0.0))
                roll = rollout(P, sur, test[:ROLLOUTS])["rollout_worst"]
                row = {"degree": q, "terms": len(G.idx), "N": N, "gradients": grad, "solves": N, "one_step_worst": float(e), "rollout_worst": float(roll),
                       "survived": survived(P, sur, test)}  # fmt: skip
                out["fits"].append(row)
                print(f"q={q} ({len(G.idx)} terms) N={N:4d} {'values+Jacobians' if grad else 'values only     '}: one-step {e:.2e}, rollout {roll:.2e}, survived {row['survived']:.2e}", flush=True)
    (OUT / "g2_hires.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(f"sensitivity solve {out['sensitivity_seconds_median'] * 1e3:.1f} ms against plain {out['plain_seconds_median'] * 1e3:.1f} ms "
          f"(ratio {out['sensitivity_over_plain']:.2f}); values agree with the benchmark's step to {check:.1e}")  # fmt: skip


if __name__ == "__main__":
    main()
