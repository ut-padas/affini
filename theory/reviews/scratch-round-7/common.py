"""Round 7 scratch: rebuilds g4_hires_recipe.py's setup so its pieces can be varied (seed of the sensitivity states, Jacobian source)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gradients"))
from g2_hires import P, poly  # noqa: E402
from g4_hires_recipe import Recipe, fit_chart, log_jacobian  # noqa: E402

from src.benchmark.step_maps import Projected, data, invariants_of, rollout, visited  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)


def fd_jacobian(z, h=1e-5):
    """Central differences of the benchmark's step: 16 plain solves."""
    E = np.eye(len(z))
    return ((P.step(z + h * E) - P.step(z - h * E)) / (2 * h)).T


def setup(seed=3, n_sens=300, jac="sens"):
    train, test = data("hires")
    X = visited(train)
    Y = train[:, 1:].transpose(1, 0, 2).reshape(-1, train.shape[2])
    mu = X.mean(axis=0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:3].T
    U = (X - mu) @ V
    dec = np.linalg.lstsq(poly(U, 4), X, rcond=None)[0]
    decode = lambda W: poly(W, 4) @ dec  # noqa: E731
    lo, hi = U.min(0), U.max(0)
    pad = 0.02 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    zlo, zhi = X.min(0) - 2.0, X.max(0) + 0.5
    W = invariants_of(P, train)
    rng = np.random.default_rng(seed)
    js = rng.choice(len(X), n_sens, replace=False)
    f = CACHE / f"J_{jac}_{seed}_{n_sens}.npy"
    if f.exists():
        Jd = np.load(f)
    else:
        Jd = np.array([(log_jacobian if jac == "sens" else fd_jacobian)(z) for z in X[js]])
        np.save(f, Jd)
    Bcoef = np.linalg.lstsq(poly(U[js], 2), Jd.reshape(len(js), -1), rcond=None)[0]
    mk = lambda G, B=None: Projected(Recipe(G, B, decode, mu, V, lo, hi, zlo, zhi), P, W)  # noqa: E731
    normal = Recipe(None, Bcoef, decode, mu, V, lo, hi, zlo, zhi).normal_term
    h = 1e-6
    dD = np.stack([(decode(U[js] + h * e) - decode(U[js] - h * e)) / (2 * h) for e in np.eye(3)], axis=2)
    targets = {"data": (Y, None), "consistent": (Y - normal(X), None), "gradients": (Y - normal(X), np.einsum("nij,njk->nik", Jd, dD))}
    return dict(train=train, test=test, X=X, Y=Y, U=U, mu=mu, V=V, lo=lo, hi=hi, zlo=zlo, zhi=zhi, W=W, js=js, Jd=Jd, Bcoef=Bcoef,
                decode=decode, mk=mk, targets=targets, rng=rng, dD=dD)  # fmt: skip


def fit(s, name, q):
    T, dT = s["targets"][name]
    return fit_chart(q, s["lo"], s["hi"], s["U"], T, s["U"][s["js"]], dT)


def jac_of(sur, Zs, h=1e-6):
    return np.stack([(sur.predict(Zs + h * e) - sur.predict(Zs - h * e)) / (2 * h) + e for e in np.eye(Zs.shape[1])], axis=2)


def spec(M):
    return np.linalg.norm(M, ord=2, axis=(1, 2))
