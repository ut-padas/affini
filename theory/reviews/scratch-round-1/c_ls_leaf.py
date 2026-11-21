"""
Review test C: is P5.1 ("a quadratic leaf holds a tolerance only if the best quadratic error is about
1 / (1 + Lambda_N) of it": 1/20 at d = 10) what happens, or only the worst case of the Lebesgue inequality?

d = 10, ART's N = 20 p + 1 = 1321 uniform points, 6 seeds, four smooth non-quadratic functions. On a fixed
evaluation set T (4000 uniform points and 1000 random vertices):
  E_T   = the best sup error on T of any quadratic (a linear program), a lower bound for the best error on the box;
  LS_int, LS_vtx = the least-squares leaf's worst error on the uniform part and on the vertices;
  the ratio LS / E_T, against 1 + Lambda_N (about 20 at the vertices, t5_least_squares.json).
Also: the pointwise identity behind the vertex errors. The LS error at x is r(x) - sum_i l_i(x) r(x_i) with r the
residual of the L2(uniform)-best quadratic; the second term has standard deviation ||r||_L2 sqrt(K(x) / N) (K the
Christoffel function, 486 at a vertex), so the vertex error is |r(vertex)| +- 0.61 ||r||_L2: the L2 projection's own
vertex error plus a seed-dependent term, with no factor Lambda_N.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "checks"))
from t5_least_squares import basis

d = 10
p = (d + 1) * (d + 2) // 2
N = 20 * p + 1
FUNCS = {
    "1/(1+0.8 mean)": lambda X: 1 / (1 + 0.8 * X.mean(1)),
    "exp(0.5 sum/sqrt d)": lambda X: np.exp(0.5 * X.sum(1) / np.sqrt(d)),
    "gaussian exp(-0.09 |x|^2)": lambda X: np.exp(-0.09 * (X * X).sum(1)),
    "cubic: mean x_i^3 + mean x_i x_i+1 x_i+2": lambda X: (X**3).mean(1) + (X * np.roll(X, -1, 1) * np.roll(X, -2, 1)).mean(1),
}
rng = np.random.default_rng(100)
T_int, T_vtx = rng.uniform(-1, 1, (4000, d)), rng.choice([-1.0, 1.0], (1000, d))
T = np.vstack([T_int, T_vtx])
BT = basis(T)
big = rng.uniform(-1, 1, (200000, d))  # for the L2(uniform)-best quadratic (orthonormal basis: coefficients are means)
Bbig = basis(big)
out = {}
for name, f in FUNCS.items():
    fT = f(T)
    # minimax on T: minimise t subject to |fT - BT c| <= t
    A = np.block([[BT, -np.ones((len(T), 1))], [-BT, -np.ones((len(T), 1))]])
    lp = linprog(np.r_[np.zeros(p), 1.0], A_ub=A, b_ub=np.r_[fT, -fT], bounds=[(None, None)] * p + [(0, None)], method="highs")
    E = float(lp.x[-1])
    c2 = Bbig.T @ f(big) / len(big)
    r_int, r_vtx = np.abs(f(T_int) - basis(T_int) @ c2).max(), np.abs(f(T_vtx) - basis(T_vtx) @ c2).max()
    r_l2 = float(np.sqrt(np.mean((f(big) - Bbig @ c2) ** 2)))
    li, lv = [], []
    for seed in range(6):
        X = np.random.default_rng(seed).uniform(-1, 1, (N, d))
        c = np.linalg.lstsq(basis(X), f(X), rcond=None)[0]
        e = np.abs(fT - BT @ c)
        li.append(float(e[:4000].max())); lv.append(float(e[4000:].max()))
    out[name] = {"E_T": E, "LS_int": li, "LS_vtx": lv, "L2proj_int": float(r_int), "L2proj_vtx": float(r_vtx), "L2_residual_norm": r_l2}
    print(f"{name:42s} E_T={E:.3e} | LS interior/E_T: {min(li)/E:.2f}-{max(li)/E:.2f} | LS vertex/E_T: {min(lv)/E:.2f}-{max(lv)/E:.2f}"
          f" | L2-projection vertex/E_T: {r_vtx/E:.2f} (interior {r_int/E:.2f}) | 0.61 ||r||_L2 / E_T = {0.61*r_l2/E:.2f}", flush=True)
print("for comparison, 1 + Lambda_N at d = 10: 11.6 (interior), 19.9 (vertices)")
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
