"""
Round-2 test D (the memo's open items for T1): Lemma 1.2 on Robertson in coordinates of the invariant leaf, against
Theorem 1 with a fixed weight vector (round-1 test B: certified degrees 19, 20, 30, 202 at tau = 0.01..10).

Coordinates z = (y1, y2, c), c = y1 + y2 + y3 (c' = 0). The Jacobian in z is
  [[-k1 - k3 y2, k3 (y3 - y2), k3 y2], [k1 + k3 y2, -k3 (y3 - y2) - 2 k2 y2, -k3 y2], [0, 0, 0]].
Lemma 1.2's row condition is imposed with equality on entrywise upper bounds over the tube |v_j| <= r_j (the majorant
system r' = G(t, r), r_c constant), along the trajectories from the box's centre and 8 corners: either one tube for
all of them (each row's largest right-hand side over the trajectories: the lemma as stated), or one tube per trajectory
(the lemma's proof is per trajectory, so r may depend on x0; only r(0) must be common). A perturbation of the initial state with |w_j| <= b_j (j = 1, 2, 3) has |w_c| <= b1 + b2 + b3, so the radii
start at r(0) = (b1, b2, b1 + b2 + b3) and the outputs obey |dy_i(tau)| <= r_i(tau), |dy_3| <= r_c + r_1 + r_2.
With b = s h (the same relative radius in every coordinate), rho = s + sqrt(s^2 + 1); the certified degree is the
smallest n for which Theorem 1's display is below 1e-3 max_B |x_i(tau)| for every output, minimised over s.
Also: the spectral abscissa of the comparison matrix of the reduced Jacobian (the smallest mu_beta over beta).
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

K1, K2, K3 = 0.04, 3e7, 1e4
rhs = lambda _t, y: [-K1 * y[0] + K3 * y[1] * y[2], K1 * y[0] - K3 * y[1] * y[2] - K2 * y[1] ** 2, K2 * y[1] ** 2]
jac = lambda _t, y: [[-K1, K3 * y[2], K3 * y[1]], [K1, -K3 * y[2] - 2 * K2 * y[1], -K3 * y[1]], [0.0, 2 * K2 * y[1], 0.0]]
YC = solve_ivp(rhs, (0, 100.0), [1.0, 0.0, 0.0], method="Radau", jac=jac, rtol=1e-12, atol=1e-20).y[:, -1]


def abscissa(J):
    C = np.abs(J); C[np.diag_indices(len(J))] = np.diag(J)
    return float(np.max(np.linalg.eigvals(C).real))


J = np.array(jac(0, YC))
y1, y2, y3 = YC
red = {"full (y1, y2, y3)": J,
       "eliminate y3": np.array([[-K1 - K3 * y2, K3 * (y3 - y2)], [K1 + K3 * y2, -K3 * (y3 - y2) - 2 * K2 * y2]]),
       "eliminate y1": np.array([[-K3 * y3 - 2 * K2 * y2 - K1, -K3 * y2 - K1], [2 * K2 * y2, 0.0]]),
       "eliminate y2": np.array([[-K1 - K3 * y3, K3 * (1 - y1 - 2 * y3)], [-2 * K2 * (1 - y1 - y3), -2 * K2 * (1 - y1 - y3)]])}
out = {"center": YC.tolist(), "comparison_abscissa": {k: abscissa(v) for k, v in red.items()}, "eigenvalues": np.sort(np.linalg.eigvals(J).real).tolist(), "cases": []}
print("y(100) =", YC, " eig J =", out["eigenvalues"])
for k, v in out["comparison_abscissa"].items():
    print(f"  inf_beta mu_beta at y(100), {k}: {v:+.4g}")


def certified_degree(s, M_over_x):
    rho = s + np.sqrt(s * s + 1)
    for n in range(1, 4000):
        if 4 * M_over_x * 3 * rho ** (-n) / (rho - 1) * ((rho + 1) / (rho - 1)) ** 2 <= 1e-3:
            return n
    return np.inf


for rel in (float(sys.argv[1]) if len(sys.argv) > 1 else 0.5,):
    h = rel * YC
    for tau in (0.01, 0.1, 1.0, 10.0):
        tg = np.concatenate([[0.0], np.logspace(-7, np.log10(tau), 200)])
        starts = [YC] + [YC + h * np.array(sg) for sg in itertools.product((-1, 1), repeat=3)]
        S = np.stack([solve_ivp(rhs, (0, tau), st, method="Radau", jac=jac, rtol=1e-11, atol=1e-18, dense_output=True).sol(tg).T for st in starts])
        xmax = np.abs(S[:, -1]).max(0)
        Y2, D = S[:, :, 1], S[:, :, 2] - S[:, :, 1]  # (trajectory, time)

        def G_one(t, r, rc, k):
            r1, r2 = r
            y2, d = np.interp(t, tg, Y2[k]), np.interp(t, tg, D[k])
            a11 = -K1 - K3 * (y2 - r2)
            a12 = K3 * (abs(d) + rc + r1 + 2 * r2)
            a1c = a2c = K3 * (y2 + r2)
            a21 = K1 + K3 * (y2 + r2)
            a22 = -K3 * (d - rc - r1 - 2 * r2) - 2 * K2 * (y2 - r2)
            return np.array([a11 * r1 + a12 * r2 + a1c * rc, a21 * r1 + a22 * r2 + a2c * rc])

        G_all = lambda t, r, rc: np.max([G_one(t, r, rc, k) for k in range(len(S))], axis=0)  # one tube for every x0 (the lemma as stated)
        blow = lambda t, r, *a: max(r) - 10.0
        blow.terminal = True

        def tube(s, per_trajectory):
            """Output radii over |x(tau)| of the tube started at b = s h, or None if the majorant blows up before tau."""
            b = s * h
            rc = b.sum()
            ends = []
            for k in (range(len(S)) if per_trajectory else [None]):
                f, args = (G_one, (rc, k)) if per_trajectory else (G_all, (rc,))
                sol = solve_ivp(f, (0, tau), [b[0], b[1]], args=args, method="Radau", rtol=1e-7, atol=1e-30, events=blow)
                if sol.status != 0 or not np.all(np.isfinite(sol.y)):
                    return None
                ends.append(sol.y[:, -1])
            r1, r2 = np.max(ends, axis=0)
            return np.array([r1, r2, rc + r1 + r2])

        row = {"rel": rel, "tau": tau}
        for label, per in (("one_tube", False), ("per_trajectory", True)):
            best, s_max = None, 0.0
            for s in np.geomspace(1e-3, 3.0, 28):
                rout = tube(s, per)
                if rout is None:
                    break
                Mx = float(np.max((xmax + rout) / xmax))
                n = certified_degree(s, Mx)
                s_max = s
                if np.isfinite(n) and (best is None or n < best["degree"]):
                    best = {"s": float(s), "rho": float(s + np.sqrt(s * s + 1)), "degree": int(n), "M_over_x": Mx, "degree_rho_only": float(np.log(1e3) / np.log(s + np.sqrt(s * s + 1)))}
            row[label] = {"s_max": float(s_max), **(best or {})}
            print(f"rel={rel} tau={tau:6} {label:14}: largest s with a tube to tau: {s_max:.3f}; best certificate: {best}", flush=True)
        out["cases"].append(row)
Path(__file__).with_name(f"d_robertson_leaf_rel{out['cases'][0]['rel']:g}.json").write_text(json.dumps(out, indent=1, default=float))
