"""
Review test A: does the tau^-1 law measured by t1_lorenz96 at F = 1 discriminate "neutral" from "dissipative"?

Same measurement as theory/checks/t1_lorenz96.py (d = 8, box [0, 2]^8, Chebyshev coefficients of Phi_tau,0 along
the x_0 fibre, rho_eff from decay_rate), repeated for F = 0.5 and 0.8 (the fixed point x = F is linearly stable:
Re lambda = -1 + F max_k(cos k - cos 2k) = -1 + F at d = 8), F = 1 (critical) and F = 1.2 (unstable, a wave), for
three fibre seeds, and for longer steps. Corollary 1.2 read as a prediction says: dissipative -> rho levels off,
neutral -> rho - 1 ~ 1/tau. Also prints mu_inf(J) on the box, the constant the theorem would actually use.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "checks"))
from common import cheb_coefficients, cheb_points, decay_rate

D, N = 8, 128
TAUS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

def flow(X, tau, F):
    def f(_t, y):
        x = y.reshape(-1, D)
        return ((np.roll(x, -1, 1) - np.roll(x, 2, 1)) * np.roll(x, 1, 1) - x + F).ravel()
    return solve_ivp(f, (0, tau), X.ravel(), method="DOP853", rtol=1e-12, atol=1e-14).y[:, -1].reshape(X.shape)

def fibre(tau, F, seed):
    base = np.random.default_rng(seed).uniform(0, 2, D)
    X = np.repeat(base[None], N + 1, 0)
    X[:, 0] = cheb_points(N, 0.0, 2.0)
    return cheb_coefficients(flow(X, tau, F)[:, 0])

out = {}
for F in (0.5, 0.8, 1.0, 1.2):
    out[F] = {}
    for seed in (1, 2, 3):
        rho = [decay_rate(fibre(t, F, seed))["rho"] for t in TAUS]
        sl = float(np.polyfit(np.log(TAUS[:5]), np.log(np.array(rho[:5]) - 1), 1)[0])
        out[F][seed] = {"rho": rho, "slope_0.25_to_4": sl}
        print(f"F={F} seed={seed} slope[0.25..4]={sl:+.2f} rho=" + " ".join(f"{r:.3g}" for r in rho), flush=True)
# the theorem's constant: mu_inf(J(x)) = -1 + 2|x_{i-1}| + |x_{i+1} - x_{i-2}| maximised over the real box [0, 2]^8
rng = np.random.default_rng(0)
X = rng.uniform(0, 2, (200000, D))
mu = np.max(-1 + 2 * np.abs(np.roll(X, 1, 1)) + np.abs(np.roll(X, -1, 1) - np.roll(X, 2, 1)), axis=1)
print("mu_inf(J) on the real box [0,2]^8: sampled max", mu.max(), "(analytic sup = 5)")
out["mu_inf_box_max_sampled"] = float(mu.max())
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float))
