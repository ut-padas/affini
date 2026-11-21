"""
Round-2 test B: Proposition 6.6's frozen-Jacobian staircase where J varies (POLLU, Robertson, the 4-D Brusselator).

At states x0 on a reference trajectory and steps tau: the singular values of DPhi_tau(x0) (variational equation, a fourth-order
Magnus product along a dense Radau solution, checked by halving its steps), in the system's variables and in log10 variables
(diag(1/x(tau)) DPhi diag(x0)), against exp(tau Re lambda_i(J(x0))). Reported: the counts above 1e-3 and 1e-6
against the frozen prediction, the largest gap in decades between a resolved singular value and its frozen
prediction, and cond(V) of the eigenvector matrix (the factor Proposition 6.6 allows).
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
warnings.filterwarnings("ignore")
from src.problems.ode.chemistry import POLLU, POLLU_Y0, ROBERTSON
from src.problems.ode.reaction_diffusion import bruss, bruss_initial


def flow(sys_, y, t):
    return solve_ivp(lambda _t, v: sys_.rhs(v), (0, t), y, method="Radau", jac=lambda _t, v: sys_.jac(v), rtol=1e-11, atol=1e-22, dense_output=True)


def dphi(sys_, x0, tau, sub=4):
    """DPhi by a fourth-order Magnus product along the dense solution, each solver step cut into `sub` pieces."""
    base = flow(sys_, x0, tau)
    ts = np.concatenate([np.linspace(a, b, sub, endpoint=False) for a, b in zip(base.t[:-1], base.t[1:])] + [[tau]])
    D = np.eye(len(x0))
    g = np.sqrt(3) / 6
    for a, b in zip(ts[:-1], ts[1:]):
        h = b - a
        A1, A2 = sys_.jac(base.sol(a + (0.5 - g) * h)), sys_.jac(base.sol(a + (0.5 + g) * h))
        D = expm(h * (A1 + A2) / 2 - np.sqrt(3) / 12 * h * h * (A1 @ A2 - A2 @ A1)) @ D
    return D, base.y[:, -1]


def case(name, sys_, y_start, t0s, taus):
    rows = []
    for t0 in t0s:
        x0 = flow(sys_, y_start, t0).y[:, -1] if t0 > 0 else y_start
        lam, V = np.linalg.eig(sys_.jac(x0))
        re = np.sort(lam.real)[::-1]
        condV = float(np.linalg.cond(V))
        Dl = np.diag(1 / x0) @ sys_.jac(x0) @ np.diag(x0)
        condV_log = float(np.linalg.cond(np.linalg.eig(Dl)[1]))
        for tau in taus:
            D, x1 = dphi(sys_, x0, tau)
            D2, _ = dphi(sys_, x0, tau, sub=8)
            s4, s8 = np.linalg.svd(D, compute_uv=False), np.linalg.svd(D2, compute_uv=False)
            conv = float(np.max(np.abs(np.log10(s4[s8 > 1e-8]) - np.log10(s8[s8 > 1e-8]))))
            D = D2
            lam1 = np.sort(np.linalg.eigvals(sys_.jac(x1)).real)[::-1]
            for coords, M in (("linear", D), ("log", np.diag(1 / x1) @ D @ np.diag(x0))):
                sig = np.linalg.svd(M, compute_uv=False)
                pred = np.exp(tau * re)
                res = sig > 1e-8
                gap = float(np.max(np.abs(np.log10(sig[res]) - np.log10(np.maximum(pred[res], 1e-300))))) if res.any() else 0.0
                rows.append({"t0": t0, "tau": tau, "coords": coords, "sigma": sig.tolist(), "frozen": pred.tolist(),
                             "count": [int(np.sum(sig > e)) for e in (1e-3, 1e-6)], "predicted": [int(np.sum(pred > e)) for e in (1e-3, 1e-6)],
                             "predicted_end": [int(np.sum(np.exp(tau * lam1) > e)) for e in (1e-3, 1e-6)],
                             "gap_decades": gap, "magnus_convergence_decades": conv, "condV": condV if coords == "linear" else condV_log})
                r = rows[-1]
                print(f"{name} t0={t0:g} tau={tau:g} {coords:6}: count={r['count']} frozen={r['predicted']} frozen_end={r['predicted_end']} gap={gap:.2f} dec condV={r['condV']:.1e}", flush=True)
                if coords == "linear":
                    print("    sigma :", " ".join(f"{s:.1e}" for s in sig[:10]))
                    print("    frozen:", " ".join(f"{s:.1e}" for s in pred[:10]))
    return rows


out = {}
out["robertson"] = case("robertson", ROBERTSON, np.array([1.0, 0, 0]), (0.01, 1.0, 100.0), (0.01, 0.1, 1.0, 10.0, 100.0))
b = bruss(2)
y0 = bruss_initial(2)(3, 0)
out["bruss2"] = case("bruss2", b, y0[0], (0.0, 2.5), (0.25, 0.5, 1.0, 2.0, 4.0))
out["pollu"] = case("pollu", POLLU, POLLU_Y0.copy(), (1.0, 10.0, 30.0), (0.01, 0.1, 1.0, 10.0))
for k, rows in out.items():
    n = len(rows)
    print(k, "count matches frozen (1e-3, 1e-6):", sum(r["count"][0] == r["predicted"][0] for r in rows), sum(r["count"][1] == r["predicted"][1] for r in rows), "of", n,
          " worst gap", max(r["gap_decades"] for r in rows))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
