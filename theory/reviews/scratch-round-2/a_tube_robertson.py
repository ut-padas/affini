"""
Round-2 test A: section 9 item 1 (Proposition 2.4 with P2.1-P2.3) on Robertson, the benchmark's setting (log10
abundances, dt = 1, 40 steps, initial states of robertson_initial), from step 1 on (past the initial layer).

Measured:
  kappa: a state on the slow manifold moved by eta dex in z2 (the fast direction) or by eta dex in every species (the
    total mass, a linear invariant); the offset that remains after one exact step. Fast-normal offset: z2 minus log10 of
    the quasi-steady y2(y1, y3); mass offset: log10 sum y.
  Two polynomial surrogates of the increment Phi(z) - z (total degree q in the three scaled log coordinates, least
  squares): V fitted on the visited states only, T on the visited states and one state per visited state drawn
  uniformly from the cube of half-width R dex around it (a tube). For each: the worst one-step error on held-out visited
  states and on held-out tube states, the worst error that survives one exact step, the worst rollout error over 39
  steps, and the chain's largest fast-normal and mass offsets.
Predictions under test: Proposition 2.4 (with accuracy eps1 on the tube the chain stays within eps1 / (1 - kappa) of
S), P2.3 (accuracy off the tube is not needed), P2.1 (rollout tracks the survived error), and the memo's reading of the
tube as "the visited states" (then V, accurate on the visited states, would roll out as well as T).
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from src.problems.ode.chemistry import ROBERTSON, robertson_initial

K1, K2, K3, DT, STEPS, R = 0.04, 3e7, 1e4, 1.0, 40, 0.05
step = lambda Z: ROBERTSON.coords_step(Z, DT)


def fast_offset(Z):
    y1, y3 = 10.0 ** Z[..., 0], 10.0 ** Z[..., 2]
    q = (-K3 * y3 + np.sqrt(K3**2 * y3**2 + 4 * K2 * K1 * y1)) / (2 * K2)
    return Z[..., 1] - np.log10(q)


mass = lambda Z: np.log10(np.sum(10.0**Z, axis=-1))


def trajectories(n, seed):
    Z = np.empty((n, STEPS + 1, 3))
    Z[:, 0] = np.log10(robertson_initial(n, seed))
    for k in range(STEPS):
        Z[:, k + 1] = step(Z[:, k])
    return Z[:, 1:]  # from step 1: past the initial layer


def powers(q):
    return np.array([(a, b, c) for a in range(q + 1) for b in range(q + 1 - a) for c in range(q + 1 - a - b)])


class Poly:
    def __init__(self, X, G, q, lo, hi):
        self.lo, self.hi, self.pw = lo, hi, powers(q)
        self.coef, _, self.rank, _ = np.linalg.lstsq(self.basis(X), G, rcond=1e-10)

    def basis(self, X):
        T = np.polynomial.chebyshev.chebvander(np.clip(2 * (X - self.lo) / (self.hi - self.lo) - 1, -3, 3), self.pw.max())  # (n, 3, q + 1)
        return T[:, 0, self.pw[:, 0]] * T[:, 1, self.pw[:, 1]] * T[:, 2, self.pw[:, 2]]

    def __call__(self, X):
        return X + self.basis(X) @ self.coef


rng = np.random.default_rng(0)
train, test = trajectories(100, 1), trajectories(40, 2)
Xv, Yv = train[:, :-1].reshape(-1, 3), train[:, 1:].reshape(-1, 3)
Xt = Xv + rng.uniform(-R, R, Xv.shape)
Yt = step(Xt)
Xe, Ye, Ye2 = test[:, :-2].reshape(-1, 3), test[:, 1:-1].reshape(-1, 3), test[:, 2:].reshape(-1, 3)
Xet = Xe + rng.uniform(-R, R, Xe.shape)
Yet = step(Xet)
lo, hi = Xv.min(0) - R, Xv.max(0) + R
out = {"box": [lo.tolist(), hi.tolist()], "visited_fast_offset_max": float(np.abs(fast_offset(Xv)).max()), "visited_mass_offset_max": float(np.abs(mass(Xv)).max())}
print("box", lo, hi, " visited states: max |fast offset|", out["visited_fast_offset_max"], " max |mass offset|", out["visited_mass_offset_max"])

# kappa
base = Xe[:: len(Xe) // 60][:60]
out["kappa"] = []
for eta in (0.01, 0.05, 0.3, 1.0):
    for sign in (1, -1):
        Pf = base.copy(); Pf[:, 1] += sign * eta
        Pm = base + sign * eta
        of, om = np.abs(fast_offset(step(Pf))).max(), np.abs(mass(step(Pm)) - mass(Pm)).max()
        tang = np.abs(step(Pf) - step(base)).max()
        out["kappa"].append({"eta": sign * eta, "fast_offset_after": float(of), "kappa_fast": float(of / eta), "tangential_shift_after": float(tang),
                             "mass_offset_change": float(om), "kappa_mass": float(np.abs(mass(step(Pm))).max() / eta)})
        print(f"eta={sign * eta:+.2f}: fast offset after one step {of:.2e} (kappa <= {of / eta:.1e}); state shift after the step {tang:.2e}; mass offset kept: kappa_mass = {out['kappa'][-1]['kappa_mass']:.6f}")

out["surrogates"] = []
for q in (3, 5):
    for name, X, G in (("V", Xv, Yv - Xv), ("T", np.vstack([Xv, Xt]), np.vstack([Yv - Xv, Yt - Xt]))):
        P = Poly(X, G, q, lo, hi)
        one_v, one_t = np.abs(P(Xe) - Ye).max(), np.abs(P(Xet) - Yet).max()
        survived = np.abs(step(P(Xe)) - Ye2).max()
        # normal derivative of the surrogate against the map's, by differences of size R / 2 at held-out visited states
        d = np.zeros(3); d[1] = R / 2
        lipn = np.abs((P(Xe + d) - P(Xe - d)) / R).max()
        lipm = np.abs((P(Xe + R / 2) - P(Xe - R / 2)) / R - 1).max()
        Y = test[:, 0].copy()
        roll, foff, moff, hist = 0.0, 0.0, 0.0, {}
        for k in range(1, STEPS):
            Y = np.clip(P(Y), -40, 5)
            err = np.abs(Y - test[:, k]).max()
            roll, foff, moff = max(roll, err), max(foff, np.abs(fast_offset(Y)).max()), max(moff, np.abs(mass(Y)).max())
            if k in (1, 2, 5, 10, 20, 39):
                hist[k] = float(err)
        row = {"q": q, "fit": name, "terms": len(P.pw), "rank": int(P.rank), "one_step_visited": float(one_v), "one_step_tube": float(one_t), "survived": float(survived),
               "rollout": float(roll), "rollout_by_step": hist, "chain_fast_offset": float(foff), "chain_mass_offset": float(moff),
               "surrogate_dz_dz2_max": float(lipn), "surrogate_mass_derivative_minus_1_max": float(lipm)}
        out["surrogates"].append(row)
        print(f"q={q} {name} (terms {row['terms']}, rank {row['rank']}): one-step visited {one_v:.2e} tube {one_t:.2e} survived {survived:.2e} rollout {roll:.2e} "
              f"(rollout/one-step {roll / one_v:.1f}, rollout/survived {roll / survived:.1f}); chain fast offset {foff:.2e} mass offset {moff:.2e}; "
              f"|dPsi/dz2| {lipn:.2f}; by step {hist}", flush=True)
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
