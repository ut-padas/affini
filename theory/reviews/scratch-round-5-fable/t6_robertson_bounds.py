"""
Checks of theory/demos/robertson_leaf.py (Demonstration 8.3), reusing its functions:
 1. deriv_bounds against a brute-force sup on 2,000,001 points (the bound must dominate), and the interval bound on |G3'| against its sampled sup;
 2. Theorem 8's defects taken where the theorem takes them, on the held-out chains' own inputs (eps_G, eps_u = max over the 40 x 39 exact
    inputs of |G(z1) - next exact state|), against the script's estimates on the reference curve;
 3. the reference's point-to-point roughness on the defect grid against the script's reference-error estimate;
 4. the bound recomputed with the chain defects, against the measured rollout.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("rl", Path(__file__).resolve().parents[2] / "demos/robertson_leaf.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)
cheb = np.polynomial.chebyshev
t0 = time.time()
ref, ref2 = rl.Curve(1e-12), rl.Curve(1e-10)
rng = np.random.default_rng(2)
y1 = rng.uniform(0.5, 1.0, 40)
y2 = 10.0 ** rng.uniform(-12, -6, 40)
S = rl.STEPS
tests = np.array([np.log10(np.maximum(rl.solve([a, b, 1 - a - b], S, 1e-12, t_eval=np.arange(S + 1.0)).y.T, 1e-300)) for a, b in zip(y1, y2, strict=True)])[:, 1:]
inputs, nexts = tests[:, :-1].reshape(-1, 3), tests[:, 1:].reshape(-1, 3)
lo, hi = inputs[:, 0].min() - rl.RHO, inputs[:, 0].max() + rl.RHO
print(f"setup {time.time() - t0:.1f} s; U_rho = [{lo:.5f}, {hi:.5f}]; z1 monotone on the curve table: {bool(np.all(np.diff(ref.z1) < 0))}")
ug = np.linspace(lo + rl.RHO, hi - rl.RHO, 20_001)
e12, e10 = ref.state(ref.time(ug) + 1.0), ref2.state(ref2.time(ug) + 1.0)
print("reference: sup |ref(1e-12) - ref(1e-10)| per component", np.abs(e12 - e10).max(axis=0), "; point-to-point jump of ref(1e-12) minus a degree-40 Chebyshev fit of itself, per component:")
sg = (2 * ug - ug[0] - ug[-1]) / (ug[-1] - ug[0])
for j in range(3):
    res = e12[:, j] - cheb.chebval(sg, cheb.chebfit(sg, e12[:, j], 40))
    print(f"   component {j}: residual sup {np.abs(res).max():.2e}, largest neighbour jump {np.abs(np.diff(res)).max():.2e}")
# held-out next states against the reference's step map at the same z1 (how well does the curve's map describe the chains?)
print("chains vs curve: sup |next state of chain - curve map at the chain's z1| per component", np.abs(ref.state(ref.time(inputs[:, 0]) + 1.0) - nexts).max(axis=0))
for q in (8, 16, 20, 24):
    s = np.cos(np.pi * np.arange(q + 1) / q)
    u = lo + (s + 1) * (hi - lo) / 2
    nxt = ref.state(ref.time(u) + 1.0)
    c1, c2 = rl.cheb_fit(nxt[:, 0], q), rl.cheb_fit(nxt[:, 1], q)
    # interpolation property of cheb_fit (orientation and normalisation)
    interp = max(np.abs(cheb.chebval(s, c1) - nxt[:, 0]).max(), np.abs(cheb.chebval(s, c2) - nxt[:, 1]).max())
    b1, b2 = rl.deriv_bounds(c1, lo, hi), rl.deriv_bounds(c2, lo, hi)
    sf = np.linspace(-1, 1, 2_000_001)
    scale = 2 / (hi - lo)
    g1, g2 = cheb.chebval(sf, c1), cheb.chebval(sf, c2)
    d1, d2 = cheb.chebval(sf, cheb.chebder(c1)) * scale, cheb.chebval(sf, cheb.chebder(c2)) * scale
    y1_, y2_ = 10.0**g1, 10.0**g2
    d3 = np.abs((y1_ * d1 + y2_ * d2) / (1 - y1_ - y2_))
    y1max, y2max = 10.0 ** b1["max"], 10.0 ** b2["max"]
    d3_bound = (y1max * b1["dsup"] + y2max * b2["dsup"]) / (1 - y1max - y2max)
    ok = b1["max"] >= g1.max() and b1["min"] <= g1.min() and b1["dsup"] >= np.abs(d1).max() and b2["dsup"] >= np.abs(d2).max() and b2["max"] >= g2.max() and d3_bound >= d3.max()
    L_g, L_G = b1["dsup"], max(b1["dsup"], b2["dsup"], d3_bound)

    def lift(v, c1=c1, c2=c2):
        sv = (2 * v - lo - hi) / (hi - lo)
        a, b = cheb.chebval(sv, c1), cheb.chebval(sv, c2)
        return np.stack([a, b, np.log10(1 - 10.0**a - 10.0**b)], axis=1)

    dchain = np.abs(lift(inputs[:, 0]) - nexts)
    eG_chain, eu_chain = float(dchain.max()), float(dchain[:, 0].max())
    dcurve = np.abs(lift(ug) - e12)
    A, bound = 0.0, 0.0
    for _ in range(1, S):
        bound = eG_chain + L_G * eu_chain * A
        A = A * L_g + 1.0
    Z, meas = tests[:, 0].copy(), 0.0
    per_comp = np.zeros(3)
    for k in range(1, S):
        Z = lift(Z[:, 0])
        meas = max(meas, float(np.abs(Z - tests[:, k]).max()))
        per_comp = np.maximum(per_comp, np.abs(Z - tests[:, k]).max(axis=0))
    print(f"q={q:2d}: interpolates its nodes to {interp:.1e}; bounds dominate brute force: {ok} (sup|g1'| {np.abs(d1).max():.6f} <= {b1['dsup']:.6f}; sup|g2'| {np.abs(d2).max():.4f} <= {b2['dsup']:.4f}; "
          f"sup|G3'| sampled {d3.max():.3f} <= interval {d3_bound:.3f})")
    print(f"      defects on the curve grid: eps_G {dcurve.max():.3e}, eps_u {dcurve[:, 0].max():.3e};  on the chains' own inputs: eps_G {eG_chain:.3e}, eps_u {eu_chain:.3e}")
    print(f"      bound with chain defects {bound:.3e}; measured rollout {meas:.3e} (per component {per_comp}); sharper L_G (sampled) would give {eG_chain + d3.max() * eu_chain * (A - 1) / L_g:.3e}")
print("seconds", round(time.time() - t0, 1))
