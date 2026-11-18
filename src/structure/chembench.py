"""
The chemistry bench: surrogates of a network's step map, trained and judged on visited states.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python -m src.structure.chembench \\
        --problems robertson --methods slowtable,chebtt_box --plot

Each problem integrates training and held-out test trajectories once (cached in
``cache/chem/<problem>.npz``). A surrogate maps a state ``z`` (log10 abundances, and log10 T for the
primordial network) to the increment ``Phi_dt(z) - z``. It is trained from the training states, in
time-major order (every trajectory's first state, then every second state, ...: the order in which a
simulation queries its cells), with access to the oracle; it is judged on every test state, against
the increments the test trajectories already contain, so testing costs no solve.

Metrics per run (``cache/chem/<problem>/<method>_tol<t>.json``): the error of an answer is the largest
|error| over the species, in dex (log10); ``worst`` is the largest over the answered test states, with
its 99.9% and 99% quantiles and the share above the tolerance; ``refused`` is the share of test states
the surrogate declines (a simulation would solve those); ``evals`` the oracle states solved in
training. Two errors are measured, since a surrogate can cover the visited states well and still drift
when chained: the one-step error above, and the rollout error of 20 test trajectories chained through
the surrogate from their initial states (the solver stepping what it refuses). ``amplification``
(``dynamics``) records how much the solver's own dynamics grow errors, to tell the two causes of drift apart:
``rollout_over_bound`` is the rollout error over the one-step worst times the accumulation factor
(near 1: errors of one sign adding up; near 1/sqrt(steps): random ones; above 1: the surrogate left the
states it was tested on). ``survived_worst`` is the part of the one-step error the next exact step
leaves (in a stiff system it damps the fast part); ``*_projected`` the same surrogate with every step
projected onto the system's linear invariants (``invariants_of``); ``speedup`` the solver's time per
state and step over the surrogate's, beside the solver's operation counts (``solve_us``, ``nfev``,
``nlu``), since SciPy's per-call overhead flatters a surrogate against a compiled solver.
``--plot`` writes ``output/chem/<problem>-errors.png`` and ``summary-chem.md``.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from src.structure import chebtt, chemistry, tabulation  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "chem"
OUT = ROOT / "output" / "chem"
YEAR = 3.156e7


@dataclass(frozen=True)
class ChemProblem:
    net: chemistry.Network
    dt: float
    initial: object  # (n, seed) -> (n, species) abundances
    steps: int
    n_train: int
    n_test: int
    tolerances: tuple  # in the problem's units (``units``)
    burn: float = 0.0  # trajectories start after this time, past the initial transient
    floor: float = 0.0  # species below this abundance are not scored (an error in dex of ~0 means nothing)

    @property
    def units(self) -> str:
        return "dex" if self.net.log else "absolute"

    def step(self, Z: np.ndarray) -> np.ndarray:
        return self.net.coords_step(Z, self.dt)

    def scored(self, Z_true: np.ndarray) -> np.ndarray:
        """Which entries count in the error: abundances above ``floor`` (the held temperature never; every entry of a linear system)."""
        if not self.net.log:
            return np.ones(Z_true.shape, dtype=bool)
        mask = Z_true > (np.log10(self.floor) if self.floor > 0 else -np.inf)
        if self.net.species[-1] == "T":
            mask[..., -1] = False
        return mask


def _primordial_initial(n: int, seed: int) -> np.ndarray:
    """n_H in [1, 1e4] cm^-3, T in [100, 5000] K, ionisation 1e-6-1e-2, H2 fraction 1e-8-1e-3 (log-uniform), He/H = 0.079."""
    rng = np.random.default_rng(seed)
    nH, T = 10.0 ** rng.uniform(0, 4, n), 10.0 ** rng.uniform(2, np.log10(5000.0), n)
    xe, xh2 = 10.0 ** rng.uniform(-6, -2, n), 10.0 ** rng.uniform(-8, -3, n)
    Y = np.full((n, 10), 1e-20) * nH[:, None]
    Y[:, 0], Y[:, 1], Y[:, 3], Y[:, 5] = nH * (1 - xe - 2 * xh2), nH * xe, nH * xh2, nH * xe
    Y[:, 6], Y[:, 9] = 0.079 * nH, T
    return Y


PROBLEMS = {
    "robertson": ChemProblem(chemistry.ROBERTSON, 1.0, chemistry.robertson_initial, 40, 200, 100, (0.01, 0.001)),
    "chemakzo": ChemProblem(chemistry.CHEMAKZO, 2.0, chemistry.chemakzo_initial, 85, 100, 40, (0.01, 0.001), burn=5.0, floor=1e-12),
    "hires": ChemProblem(chemistry.HIRES, 2.0, chemistry.hires_initial, 170, 100, 40, (0.01, 0.001), burn=5.0, floor=1e-12),
    "primordial": ChemProblem(chemistry.PRIMORDIAL, 1e4 * YEAR, _primordial_initial, 30, 60, 30, (0.01, 0.001), floor=1e-12),
    "pollu": ChemProblem(chemistry.POLLU, 1.0, chemistry.pollu_initial, 59, 100, 40, (0.01, 0.001), burn=1.0, floor=1e-12),
}  # fmt: skip (in order of input dimension: 3, 5, 8, 10, 20)
# The stiff ladder: each family of chemistry.LADDER at every size, named <family>_d<input dimension>.
_LADDER_STEP = {"bruss": (0.25, 40), "allen_cahn": (0.005, 40), "fisher_kpp": (0.01, 40)}  # (dt, steps): 40 steps that still move
for _family, (_make, _initial, _sizes) in chemistry.LADDER.items():
    for _n in _sizes:
        _net = chemistry.NETWORKS[_make(_n).name]
        PROBLEMS[f"{_family}_d{len(_net.species)}"] = ChemProblem(_net, _LADDER_STEP[_family][0], _initial(_n), _LADDER_STEP[_family][1],
                                                                  100, 40, (0.01, 0.001))  # fmt: skip


# The solvers a surrogate's speed is measured against. A speedup is always "over" one of them, named
# with its integrator, settings and hardware, so that a later baseline (a compiled or GPU integrator)
# adds a record and every plot can be redrawn against it, without retraining anything.
BASELINES = {
    "scipy-radau-cpu1": {
        "label": "SciPy Radau, 1 CPU core, one state at a time",
        "integrator": "scipy.integrate.solve_ivp(method='Radau'): 3-stage Radau IIA (order 5), analytic Jacobian",
        "settings": "each network's rtol (1e-8) and atol; one call per state and step",
        "device": "CPU, 1 core (OPENBLAS_NUM_THREADS=1)",
        "note": "SciPy's per-call Python overhead dominates small systems; nfev and nlu give the work a compiled solver would do",
    },
}
BASELINE = "scipy-radau-cpu1"  # the default for plots and the recorded ``speedup``


def baseline_cost(name: str, baseline: str = BASELINE) -> dict:
    """
    One baseline's cost of a step per state on 40 test states (cached in ``<problem>-baseline-<baseline>.json``):
    wall time in microseconds and the operation counts (right-hand sides, Jacobians, LU factorisations), with the
    baseline's description. Only SciPy's Radau is measured here; another baseline's records are written by its
    own runner in the same form.
    """
    path = CACHE / f"{name}-baseline-{baseline}.json"
    if path.exists():
        return json.loads(path.read_text())
    if baseline != "scipy-radau-cpu1":
        raise KeyError(f"no measurement of baseline {baseline!r} for {name}")
    old = CACHE / f"{name}-cost.json"  # the same measurement, recorded before baselines were named
    if old.exists():
        cost = json.loads(old.read_text())
    else:
        p = PROBLEMS[name]
        _, test = data(name)
        Z = test[:, :-1].reshape(-1, test.shape[2])
        Z = Z[np.random.default_rng(0).choice(len(Z), min(40, len(Z)), replace=False)]
        cost = p.net.step_cost(10.0**Z if p.net.log else Z, p.dt)
    cost = {**cost, "baseline": baseline, **BASELINES[baseline]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cost))
    return cost


def solver_cost(name: str) -> dict:
    """The default baseline's cost (``baseline_cost``)."""
    return baseline_cost(name, BASELINE)


def speedup(m: dict, name: str, baseline: str = BASELINE) -> float:
    """A run's speedup over a named baseline: the baseline's time per state and step over the surrogate's inference time per state."""
    return baseline_cost(name, baseline)["solve_us"] / max(m["predict_us"], 1e-9)


def data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) trajectories, (n, steps + 1, d) states in the surrogates' coordinates, integrated once and cached."""
    path = CACHE / f"{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["train"], z["test"]
    p = PROBLEMS[name]
    start = [p.initial(n, seed) for n, seed in ((p.n_train, 1), (p.n_test, 2))]
    if p.burn > 0:
        start = [p.net.step(Y, p.burn) for Y in start]
    train, test = (chemistry.trajectories(p.net, Y, p.dt, p.steps) for Y in start)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train=train, test=test)
    return train, test


class CachedOracle:
    """The step map with a cache by exact state, counting distinct solves (vector outputs: one solve serves every species)."""

    def __init__(self, step):
        self.step, self.cache, self.evals = step, {}, 0

    def __call__(self, Z):
        Z = np.atleast_2d(Z)
        keys = [z.tobytes() for z in Z]
        new = [i for i, k in enumerate(keys) if k not in self.cache]
        if new:
            out = self.step(Z[new])
            self.evals += len(new)
            self.cache.update({keys[i]: v for i, v in zip(new, out, strict=True)})
        return np.array([self.cache[k] for k in keys])


def visited(train: np.ndarray) -> np.ndarray:
    """The training trajectories' states that start a step, in time-major order (every trajectory's first state, then every second, ...)."""
    return train[:, :-1].transpose(1, 0, 2).reshape(-1, train.shape[2])


def slowtable(p: ChemProblem, train: np.ndarray, atol: float, seed: int):
    oracle = CachedOracle(p.step)
    t = tabulation.build(oracle, visited(train), atol, seed=seed)
    t.evals = oracle.evals
    return t


class BoxTT:
    """One ChebTT per output component on the padded bounding box of the training states."""

    def __init__(self, models, lo, hi, evals):
        self.models, self.lo, self.hi, self.evals = models, lo, hi, evals

    def predict(self, Q):
        Q = np.clip(Q, self.lo, self.hi)
        return np.stack([m.predict(Q) if m is not None else np.zeros(len(Q)) for m in self.models], axis=1)

    def refuses(self, Q):
        return np.any((self.lo > Q) | (self.hi < Q), axis=1)


def state_box(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The visited states' bounding box padded by 2% per axis, and widened to 2e-3 on an axis the data never varies."""
    lo, hi = states.min(axis=0), states.max(axis=0)
    pad = 0.02 * np.maximum(hi - lo, 1e-3)
    lo, hi = lo - pad, hi + pad
    flat = hi - lo <= 2e-3
    return np.where(flat, lo - 1e-3, lo), np.where(flat, hi + 1e-3, hi)


def chebtt_box(p: ChemProblem, train: np.ndarray, atol: float, seed: int, oracle: CachedOracle | None = None, edges: bool = True):
    oracle = oracle or CachedOracle(p.step)
    lo, hi = state_box(visited(train))
    models = []
    for c in range(train.shape[2]):
        if p.net.species[c] == "T":
            models.append(None)  # the held temperature: increment 0
            continue
        models.append(chebtt.build(lambda X, c=c: (oracle(X) - X)[:, c], lo, hi, atol, seed=seed, max_evals=200_000, edges=edges))
    return BoxTT(models, lo, hi, oracle.evals)


def chebtt_smooth(p: ChemProblem, train: np.ndarray, atol: float, seed: int):
    """``chebtt_box`` without the edge search: one Chebyshev basis per axis (the control for piecewise axes)."""
    return chebtt_box(p, train, atol, seed, edges=False)


def chebtt_adaptive_box(p: ChemProblem, train: np.ndarray, atol: float, seed: int):
    """ChebTT-A (``chebtt_adaptive``) per species on the states' bounding box: piecewise axes, and an interface term where one is found."""
    from src.structure import chebtt_adaptive

    oracle = CachedOracle(p.step)
    lo, hi = state_box(visited(train))
    models = [None if p.net.species[c] == "T" else
              chebtt_adaptive.build(lambda X, c=c: (oracle(X) - X)[:, c], lo, hi, atol, seed=seed, max_evals=200_000)
              for c in range(train.shape[2])]  # fmt: skip
    return BoxTT(models, lo, hi, oracle.evals)


class BoxBaseline(BoxTT):
    """A box-bench baseline (AFINNI or ART, ``bench.METHODS``) of every species' increment on the states' bounding box."""

    def __init__(self, model, lo, hi, evals, outputs: int, held: bool):
        super().__init__(model, lo, hi, evals)
        self.outputs, self.held = outputs, held

    def predict(self, Q):
        Q = np.clip(Q, self.lo, self.hi)
        out = np.asarray(self.models.predict(Q), dtype=np.float64).reshape(len(Q), self.outputs)
        return np.concatenate([out, np.zeros((len(Q), 1))], axis=1) if self.held else out


def baseline(name: str):
    """A stiff-bench method running the box bench's ``name`` (AFINNI, ART) with the solver as its oracle."""

    def method(p: ChemProblem, train: np.ndarray, atol: float, seed: int):
        from src import to_host, xp
        from src.structure import bench

        held = p.net.species[-1] == "T"
        m = train.shape[2] - (1 if held else 0)
        lo, hi = state_box(visited(train))
        oracle = CachedOracle(p.step)

        def f(X):
            Z = np.asarray(to_host(X), dtype=np.float64)
            return xp.asarray((oracle(Z) - Z)[:, :m])

        model = bench.METHODS[name][2](f, lo, hi, atol, seed)
        return BoxBaseline(model, lo, hi, oracle.evals, m, held)

    return method


def chebtt_roll(p: ChemProblem, train: np.ndarray, atol: float, seed: int, rounds: int = 3):
    """
    ``chebtt_box``, accepted only once the surrogate chained along every training trajectory (projected
    onto the invariants, as it is used) stays within half of ``atol`` of them (a margin for the test
    trajectories, which may pass nearer a sensitive state: Allen-Cahn's unstable u = 0); else retrained
    at the one-step tolerance times the shortfall (at most 8x tighter per round, 0.8 for margin). The training trajectories are known, so the check costs no solve; the retraining reuses
    every cached solve, and ChebTT's degrees grow only logarithmically with the accuracy.
    """
    oracle = CachedOracle(p.step)
    W = invariants_of(p, train)
    one = atol
    for _ in range(rounds):
        model = chebtt_box(p, train, one, seed, oracle)
        used = Projected(model, p, W) if len(W) else model
        worst = rollout(p, used, train)["rollout_worst"]
        if worst <= 0.5 * atol:
            break
        one *= 0.8 * max(0.5 * atol / worst, 0.125)
    model.evals, model.one_step_tol = oracle.evals, one
    return model


class BoxVectorTT(BoxTT):
    """``chebtt.build_vector`` (rotated per output group where that pays) on the padded bounding box of the training states."""

    def predict(self, Q):
        Q = np.clip(Q, self.lo, self.hi)
        out = self.models.predict(Q)
        return np.concatenate([out, np.zeros((len(Q), 1))], axis=1) if self.held else out


def chebtt_metric(p: ChemProblem, train: np.ndarray, atol: float, seed: int):
    states = visited(train)
    held = p.net.species[-1] == "T"  # the held temperature is an input only: its increment is 0
    m = states.shape[1] - (1 if held else 0)
    lo, hi = states.min(axis=0), states.max(axis=0)
    pad = 0.02 * np.maximum(hi - lo, 1e-3)
    lo, hi = lo - pad, hi + pad
    oracle = CachedOracle(p.step)
    model = chebtt.build_vector(lambda X: (oracle(X) - X)[:, :m], lo, hi, atol, m, rotate=True, seed=seed, max_evals=200_000)
    out = BoxVectorTT(model, lo, hi, oracle.evals)
    out.held = held
    return out


METHODS = {"slowtable": ("SlowTable: tabulated visited states", "#8e24aa", slowtable),
           "chebtt_smooth": ("ChebTT, one basis per axis (no edge search)", "#c9a227", chebtt_smooth),
           "chebtt_box": ("ChebTT, piecewise axes", "#e08a00", chebtt_box),
           "chebtt_adaptive": ("ChebTT-A: piecewise axes and interface terms", "#b5179e", chebtt_adaptive_box),
           "chebtt_metric": ("ChebTT in the pullback-metric coordinates", "#1b7f5b", chebtt_metric),
           "chebtt_roll": ("ChebTT, retrained until its rollouts hold", "#c0392b", chebtt_roll),
           "afinni": ("AFINNI", "#6d3b0f", baseline("afinni")),
           "art": ("ART, affine leaves", "#1f5fa8", baseline("art")),
           "art_quad": ("ART, quadratic leaves", "#2e7d32", baseline("art_quad"))}  # fmt: skip


def invariants_of(p: ChemProblem, train: np.ndarray) -> np.ndarray:
    """
    (m, d) the system's linear invariants W (W F(y) = 0 for every y: element and charge totals of a
    network, total mass), from the null space of the right-hand side at up to 600 training states (the
    held temperature excluded), rows orthonormal; empty when there are none. In the system's own
    variables (abundances, not their log10).
    """
    Z = visited(train)
    Z = Z[np.random.default_rng(0).choice(len(Z), min(600, len(Z)), replace=False)]
    free = [i for i in range(Z.shape[1]) if p.net.species[i] != "T"]
    Y = 10.0**Z if p.net.log else Z
    F = np.array([np.asarray(p.net.rhs(y))[free] for y in Y])
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-300)
    _, sv, Vt = np.linalg.svd(F, full_matrices=True)
    sv = np.concatenate([sv, np.zeros(len(free) - len(sv))])
    null = Vt[sv < 1e-9 * max(sv.max(), 1e-300)]
    W = np.zeros((len(null), Z.shape[1]))
    W[:, free] = null
    return W


class Projected:
    """
    A surrogate whose every step is projected onto the invariants of its input state, ``W y' = W y``:
    in linear coordinates the nearest such state; in log coordinates the smallest relative change,
    ``y' = y_hat + D^2 W^T (W D^2 W^T)^-1 (W y - W y_hat)`` with ``D = diag(y_hat)`` (each abundance
    moves in proportion to itself, so trace species are barely touched), floored at ``FLOOR``.
    """

    def __init__(self, model, p: ChemProblem, W: np.ndarray):
        self.model, self.p, self.W = model, p, W
        self.evals = getattr(model, "evals", 0)

    def refuses(self, Z):
        return self.model.refuses(Z)

    def predict(self, Z):
        inc = self.model.predict(Z)
        log = self.p.net.log
        y, yh = (10.0**Z, 10.0 ** (Z + inc)) if log else (Z, Z + inc)
        D2 = yh * yh if log else np.ones_like(yh)
        r = (y - yh) @ self.W.T  # (N, m) the invariants' defect
        M = np.einsum("ij,nj,kj->nik", self.W, D2, self.W)
        ys = yh + D2 * np.einsum("nm,mj->nj", np.linalg.solve(M, r[:, :, None])[:, :, 0], self.W)
        return (np.log10(np.maximum(ys, chemistry.FLOOR)) if log else ys) - Z


def run(method: str, name: str, tol: float, seed: int = 1, force: bool = False) -> dict:
    path = CACHE / name / f"{method}_tol{tol:g}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    p = PROBLEMS[name]
    train, test = data(name)
    states = train[:, :-1].transpose(1, 0, 2).reshape(-1, train.shape[2])  # time-major
    Zt, Gt = test[:, :-1].reshape(-1, test.shape[2]), (test[:, 1:] - test[:, :-1]).reshape(-1, test.shape[2])
    t0 = time.perf_counter()
    try:
        model = METHODS[method][2](p, train, tol, seed)
    except RuntimeError as err:
        m = {"method": method, "problem": name, "tol": tol, "failed": str(err)[:300]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(m))
        return m
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Zt)
    predict_us = 1e6 * (time.perf_counter() - t0) / len(Zt)
    refused = np.asarray(model.refuses(Zt), dtype=bool)
    mask = p.scored(test[:, 1:].reshape(-1, test.shape[2]))
    e = np.max(np.where(mask, np.abs(pred - Gt), 0.0), axis=1)
    ea = e[~refused] if (~refused).any() else np.zeros(1)
    m = {"method": method, "problem": name, "tol": tol, "evals": int(model.evals), "train_states": len(states),
         "worst": float(ea.max()), "q999": float(np.quantile(ea, 0.999)), "q99": float(np.quantile(ea, 0.99)),
         "above_percent": float(100 * np.mean(ea > tol)), "refused_percent": float(100 * refused.mean()),
         "train_s": train_s, "predict_us": predict_us, "records": len(getattr(model, "Z", [])) or None,
         "error_by_step": np.max(np.where(refused, 0.0, e).reshape(test.shape[0], -1), axis=0).tolist(),
         "refused_by_step": refused.reshape(test.shape[0], -1).mean(axis=0).tolist()}  # fmt: skip
    m.update(rollout(p, model, test[:ROLLOUTS]))
    m.update(solver_cost(name))
    acc = np.asarray(dynamics(name)["accumulation"])
    roll = np.asarray(m["rollout_error_by_step"])
    # from step 2: at step 1 the rollout error is the one-step error itself, so the ratio is 1 by construction
    m["rollout_over_bound"] = float(np.max(roll[1:] / (max(m["worst"], 1e-300) * acc[1 : len(roll)]))) if len(roll) > 1 else 0.0
    m["survived_worst"] = survived(p, model, test)
    if hasattr(model, "one_step_tol"):
        m["one_step_tol"] = model.one_step_tol
    W = invariants_of(p, train)
    m["invariants"] = len(W)
    if len(W):  # the same surrogate, every step projected onto the input's invariants
        proj = Projected(model, p, W)
        ep = np.max(np.where(mask, np.abs(proj.predict(Zt) - Gt), 0.0), axis=1)[~refused]
        r = rollout(p, proj, test[:ROLLOUTS])
        m.update({"worst_projected": float(ep.max()) if len(ep) else 0.0, "rollout_worst_projected": r["rollout_worst"],
                  "rollout_error_by_step_projected": r["rollout_error_by_step"]})  # fmt: skip
    m["speedup"] = m["solve_us"] / max(predict_us, 1e-9)  # over BASELINE: the solver's time per state and step over the surrogate's
    m["inference"] = {"device": "cpu", "threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or 0), "batch": len(Zt),
                      "baseline": BASELINE}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m))
    return m


ROLLOUTS = 20  # test trajectories chained through each surrogate


def survived(p: ChemProblem, model, test: np.ndarray, n: int = 200) -> float:
    """
    The worst one-step error that survives the next exact step, ``|Phi(S(z)) - Phi(Phi(z))|`` over n
    answered test states (one solve each; ``Phi(Phi(z))`` is the trajectory's state two steps on):
    in a stiff system the next step damps the error's fast part, and what is left is what a
    simulation carries forward.
    """
    Z = test[:, :-2].reshape(-1, test.shape[2])
    Z2 = test[:, 2:].reshape(-1, test.shape[2])
    idx = np.flatnonzero(~np.asarray(model.refuses(Z), dtype=bool))
    if not len(idx):
        return 0.0
    idx = idx[np.random.default_rng(0).choice(len(idx), min(n, len(idx)), replace=False)]
    out = p.step(Z[idx] + model.predict(Z[idx]))
    return float(np.max(np.where(p.scored(Z2[idx]), np.abs(out - Z2[idx]), 0.0)))


def rollout(p: ChemProblem, model, true: np.ndarray) -> dict:
    """
    Chain the surrogate from each true initial state for every step of the trajectory, the solver
    stepping the states it refuses (as a simulation falls back), and compare with the solver's
    trajectory: the error a one-zone run accumulates, which the one-step error alone does not show.
    """
    Z = true[:, 0].copy()
    err, solved = [], []
    for k in range(true.shape[1] - 1):
        r = np.asarray(model.refuses(Z), dtype=bool)
        nxt = Z + model.predict(Z)
        if r.any():
            nxt[r] = p.step(Z[r])
        Z = nxt
        err.append(float(np.max(np.where(p.scored(true[:, k + 1]), np.abs(Z - true[:, k + 1]), 0.0))))
        solved.append(float(r.mean()))
    return {"rollout_error_by_step": err, "rollout_solved_by_step": solved, "rollout_worst": max(err)}


def dynamics(name: str) -> dict:
    """
    What the dynamics alone do to errors, from the ``ROLLOUTS`` test trajectories (cached), the same for
    every surrogate:

    - ``stiffness_index``: dt times the fastest decay rate max |Re lambda| of the right-hand side's
      Jacobian (median over the states): how many e-folds the fastest mode decays within one step.
      ``stiffness_ratio``: that rate over the slowest nonzero one. ``step_stiffness``: the same read
      off the step map, max over the steps of -ln min |eig J_k| (Jay and Calvo: mildly stiff from 10,
      stiff from 20), which ``kind`` uses.
    - ``growth``: (steps,) the largest singular value of ``J_k ... J_1`` (J the step map's Jacobian
      in the surrogates' coordinates, central differences): an error made at the first step after k
      steps. ``lyapunov``: log growth per unit time at the end (finite-time Lyapunov exponent).
    - ``accumulation``: (steps,) ``A_n = sum_{j=1..n} |J_{n-1} ... J_j|`` (the identity for j = n),
      the classical global-error bound's factor: one-step errors of at most e give a rollout error of
      at most about ``e A_n`` after n steps, whatever the surrogate.
    - ``kind``: "chaotic" when the growth exceeds 10 by the end with a positive exponent, "contracting"
      when it ends below 1, else "neutral" (oscillatory, conserving or slowly drifting: errors along
      such directions neither grow nor decay); and "stiff" or "mildly stiff" by ``step_stiffness``.
    """
    path = CACHE / f"{name}-dynamics.json"
    if path.exists():
        return json.loads(path.read_text())
    p = PROBLEMS[name]
    _, test = data(name)
    true = test[:ROLLOUTS]  # the rollouts' own trajectories, so their errors and A_n refer to the same states
    n_steps, d = true.shape[1] - 1, true.shape[2]
    free = [i for i in range(d) if p.net.species[i] != "T"]
    h = 1e-5
    E = np.zeros((2 * len(free), d))
    for j, i in enumerate(free):
        E[2 * j, i], E[2 * j + 1, i] = h, -h
    growth, acc, fast, slow, step_stiff = np.zeros((len(true), n_steps)), np.zeros((len(true), n_steps)), [], [], []
    for t, traj in enumerate(true):
        Js = []
        for k in range(n_steps):
            out = p.step(traj[k][None] + E)[:, free]
            Js.append((out[0::2] - out[1::2]).T / (2 * h))
            y = 10.0 ** traj[k] if p.net.log else traj[k]
            lam = np.abs(np.real(np.linalg.eigvals(np.asarray(p.net.jac(y))[np.ix_(free, free)])))
            fast.append(lam.max())
            slow.append(lam[lam > 1e-9 * lam.max()].min())
        step_stiff.append(max(-np.log(max(float(np.min(np.abs(np.linalg.eigvals(J)))), 1e-300)) for J in Js))
        P = np.eye(len(free))
        for k, J in enumerate(Js):
            P = J @ P
            growth[t, k] = np.linalg.norm(P, 2)
        for n in range(1, n_steps + 1):  # A_n: an error made after step j is carried by J_{n-1} ... J_j
            S, total = np.eye(len(free)), 1.0
            for j in range(n - 1, 0, -1):
                S = S @ Js[j]
                total += np.linalg.norm(S, 2)
            acc[t, n - 1] = total
    g = growth.max(axis=0)
    out = {"stiffness_index": float(p.dt * np.median(fast)), "stiffness_ratio": float(np.median(np.array(fast) / np.array(slow))),
           "step_stiffness": float(np.max(step_stiff)),
           "growth": g.tolist(), "accumulation": acc.max(axis=0).tolist(),
           "lyapunov": float(np.log(max(g[-1], 1e-300)) / (n_steps * p.dt))}  # fmt: skip
    motion = "chaotic" if g[-1] > 10 and out["lyapunov"] > 0 else "contracting" if g[-1] < 1 else "neutral"
    st = out["step_stiffness"]
    out["kind"] = motion + (", stiff" if st >= 20 else ", mildly stiff" if st >= 10 else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def plot(name: str, methods: list[str]) -> Path:
    """
    Four panels over the steps of the test trajectories: the one-step error (a surrogate step from the
    true state against the solver's step: how well the visited states are covered), the rollout error
    (the surrogate chained from the initial state, the solver stepping what it refuses, against the
    solver's trajectory), the share of states the surrogate refuses, and the growth the dynamics alone
    allow an early error (the same for every surrogate).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    (a, b), (c, g) = axes
    tols = sorted(PROBLEMS[name].tolerances, reverse=True)
    handles = []
    for method in methods:
        solves = []
        for tol, ls in zip(tols, ("-", "--"), strict=False):
            path = CACHE / name / f"{method}_tol{tol:g}.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            if "failed" in m:
                solves.append("over")
                continue
            solves.append(f"{m['evals']:,}")
            colour = METHODS[method][1]
            a.semilogy(np.maximum(m["error_by_step"], 1e-12) / tol, ls, color=colour, lw=1.3)
            if "rollout_error_by_step" in m:
                b.semilogy(np.maximum(m["rollout_error_by_step"], 1e-12) / tol, ls, color=colour, lw=1.3)
            c.plot(100 * np.array(m["refused_by_step"]), ls, color=colour, lw=1.3)
        if solves:  # one entry per method: its colour, and its training solves at each tolerance
            label = f"{SHORT.get(method, method).replace(chr(10), ' ')} ({' / '.join(solves)} solves)"
            handles.append(matplotlib.lines.Line2D([], [], color=METHODS[method][1], lw=2, label=label))
    handles += [matplotlib.lines.Line2D([], [], color="#555555", ls=ls, lw=1.3, label=f"tol {t:g} {PROBLEMS[name].units}")
                for t, ls in zip(tols, ("-", "--"), strict=False)]  # fmt: skip
    try:
        dyn = dynamics(name)
        g.semilogy(dyn["growth"], color="#444444", label="growth of an error made at step 1")
        g.semilogy(dyn["accumulation"], color="#888888", ls="--", label="accumulation A_n (bound factor for one-step errors)")
        g.legend(fontsize=7, framealpha=0.85, loc="best")
        g.set_title(f"what the dynamics do to errors ({dyn['kind']}; stiffness index {dyn['stiffness_index']:.3g}, "
                    f"FTLE {dyn['lyapunov']:.2g})", fontsize=9)
    except Exception as err:  # noqa: BLE001 (the panel is optional; a failed solve must not lose the others)
        g.text(0.5, 0.5, f"not computed: {err}", transform=g.transAxes, ha="center", fontsize=8)
    for ax in (a, b):
        ax.axhline(1.0, color="#c0392b", lw=0.8, ls=":")
    g.axhline(1.0, color="#999999", lw=0.8, ls=":")
    a.set_title("one-step error: a surrogate step from the true state vs the solver's step", fontsize=9)
    b.set_title("rollout error: the surrogate chained from the initial state vs the solver's trajectory", fontsize=9)
    c.set_title("states the surrogate refuses (the solver steps them)", fontsize=9)
    a.set_ylabel("worst |error| / tolerance (answered states)")
    b.set_ylabel("worst |error| / tolerance over 20 rollouts")
    c.set_ylabel("% of test states refused")
    g.set_ylabel("factor (same for every surrogate)")
    for ax in axes.ravel():
        ax.set_xlabel("step of the test trajectories")
    fig.legend(handles=handles, loc="outside lower center", ncol=min(5, len(handles)), fontsize=8, frameon=False)
    p = PROBLEMS[name]
    units = "log10 (dex)" if p.net.log else "absolute units"
    try:
        cost = solver_cost(name)
        solver = f"; the solver: {cost['solve_us']:,.0f} us per state and step ({cost['nfev']:.0f} RHS, {cost['nlu']:.0f} LU)"
    except Exception:  # noqa: BLE001 (the cost is a caption only)
        solver = ""
    fig.suptitle(f"{name}: surrogates of the step map (dt = {p.dt:.3g}), on held-out trajectories; errors in {units}{solver}", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}-errors.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ladder() -> Path:
    """
    ``output/chem/ladder.png``: per family (columns; the fixed-size test problems together), against the
    input dimension, the one-step worst error, the part of it that survives the next exact step and the
    rollout worst error (all over the tolerance), the training solves, and the speedup (the solver's time per state and step over the surrogate's). One
    colour per method, solid for the looser tolerance and dashed for the tighter; an x at the top of
    the error panels marks a case over the time limit or the evaluation budget.
    """
    from src.structure.chemfill import FIXED, family, ladder

    names = ladder(10**6)
    fams = [f for f in dict.fromkeys(family(n) for n in names) if f not in FIXED] + ["test set"]
    rows = (("worst", "one-step worst / tol"), ("survived_worst", "one-step error surviving the next step / tol"),
            ("rollout_worst", "rollout worst / tol"), ("evals", "training solves"), ("speedup", "solver time / inference time"))  # fmt: skip
    fig, axes = plt.subplots(len(rows), len(fams), figsize=(4.2 * len(fams), 13.5), squeeze=False, layout="constrained")
    for j, fam in enumerate(fams):
        members = [n for n in names if (family(n) == fam) or (fam == "test set" and n in FIXED)]
        for method, (label, colour, _) in METHODS.items():
            for k, ls in enumerate(("-", "--")):
                pts, fails = {key: [] for key, _ in rows}, []
                for n in members:
                    p = PROBLEMS[n]
                    if k >= len(p.tolerances):
                        continue
                    tol = sorted(p.tolerances, reverse=True)[k]
                    path = CACHE / n / f"{method}_tol{tol:g}.json"
                    if not path.exists():
                        continue
                    m = json.loads(path.read_text())
                    d = len(p.net.species)
                    if "failed" in m:
                        fails.append(d)
                        continue
                    for key, _ in rows:
                        v = m.get(key)
                        if v is not None:
                            pts[key].append((d, v / tol if key in ("worst", "survived_worst", "rollout_worst") else v))
                for i, (key, _) in enumerate(rows):
                    if pts[key]:
                        x, y = zip(*sorted(pts[key]), strict=True)
                        axes[i, j].plot(x, y, ls, marker="o", ms=3.5, color=colour)
                for i in (0, 1, 2):
                    for d in fails:
                        axes[i, j].plot(d, 30.0, "x", color=colour, ms=7)
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            ax.set_yscale("log")
            ax.set_xscale("log")
            if i < 3:
                ax.axhline(1.0, color="#c0392b", lw=0.8, ls=":")
            ax.set_ylabel(ylabel if j == 0 else "")
            ax.set_xlabel("input dimension" if i == len(rows) - 1 else "")
            dims = sorted({len(PROBLEMS[n].net.species) for n in members})
            if dims:
                ax.set_xticks(dims, [str(d) for d in dims])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        title = fam if fam != "test set" else "Robertson (3), Akzo Nobel (5), HIRES (8), POLLU (20)"
        axes[0, j].set_title(title, fontsize=10)
    shown = [m for m in METHODS if any((CACHE / n / f"{m}_tol{t:g}.json").exists() for n in names for t in PROBLEMS[n].tolerances)]
    handles = [matplotlib.lines.Line2D([], [], color=METHODS[m][1], lw=2, marker="o", ms=4, label=SHORT.get(m, m).replace(chr(10), " ")) for m in shown]
    handles += [matplotlib.lines.Line2D([], [], color="#555555", ls=ls, label=f"{w} tolerance") for w, ls in (("looser", "-"), ("tighter", "--"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(6, len(handles)), fontsize=8, frameon=False)
    fig.suptitle("Stiff-system ladder: surrogates of the step map on held-out trajectories (x at the top: over the time limit or budget)",
                 fontsize=11)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ladder.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


SHORT = {"slowtable": "SlowTable", "chebtt_smooth": "ChebTT\nno edges", "chebtt_box": "ChebTT", "chebtt_adaptive": "ChebTT-A",
         "chebtt_metric": "ChebTT-M", "chebtt_roll": "ChebTT\nrollout", "afinni": "AFINNI", "art": "ART\naffine", "art_quad": "ART\nquad"}  # fmt: skip


def plot_overview(baseline: str = BASELINE) -> list[Path]:
    """
    ``output/chem/overview-stiff-{loose,tight}.png``: per problem (rows, by input dimension) and method
    (columns), the worst one-step error and the worst rollout error over the tolerance (blue within,
    red beyond), with the training solves; hatched: over the time limit or the evaluation budget.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from src.structure.chemfill import ladder

    names = [n for n in ladder(10**6) if (CACHE / n).is_dir()]
    methods = [m for m in METHODS if any((CACHE / n / f"{m}_tol{t:g}.json").exists() for n in names for t in PROBLEMS[n].tolerances)]
    bounds = [0, 0.01, 0.1, 0.5, 1, 2, 10, 100, 1e300]
    cmap = ListedColormap(["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f4a582", "#d6604d", "#b2182b", "#67001f"])
    norm = BoundaryNorm(bounds, cmap.N)
    paths = []
    for k, label in ((0, "loose"), (1, "tight")):
        fig, axes = plt.subplots(3, 1, figsize=(0.78 * len(methods) + 2.6, 3 * (0.34 * len(names) + 1.45)), constrained_layout=True)
        # third panel: speedup over the named baseline, in decade bands of a perceptually uniform,
        # colour-blind-safe scale (viridis), the row's fastest boxed
        decades = [(1e1, "< 10x"), (1e2, "10-100x"), (1e3, "100-1,000x"), (1e4, "1,000-10,000x"), (np.inf, ">= 10,000x")]
        band = [plt.get_cmap("viridis")(v) for v in (0.0, 0.3, 0.55, 0.75, 0.97)]
        for i, n in enumerate(names):
            tols = sorted(PROBLEMS[n].tolerances, reverse=True)
            if k >= len(tols):
                continue
            row = []
            for j, meth in enumerate(methods):
                path = CACHE / n / f"{meth}_tol{tols[k]:g}.json"
                if not path.exists():
                    continue
                m = json.loads(path.read_text())
                if "failed" in m:
                    axes[2].add_patch(plt.Rectangle((j, i), 0.95, 0.9, fc="#e8e8e8", hatch="//", ec="#999999"))
                    continue
                try:
                    sp = speedup(m, n, baseline)
                except KeyError:
                    continue
                b = next(q for q, (top, _) in enumerate(decades) if sp < top)
                axes[2].add_patch(plt.Rectangle((j, i), 0.95, 0.9, fc=band[b], ec="white"))
                axes[2].text(j + 0.47, i + 0.45, f"{sp:,.0f}x\n{m['predict_us']:.2g} us", ha="center", va="center", fontsize=6.5,
                             color="white" if b <= 1 else "#111111")  # fmt: skip
                row.append((sp, j))
            if row:
                jb = max(row)[1]
                axes[2].add_patch(plt.Rectangle((jb, i), 0.95, 0.9, fill=False, ec="black", lw=1.6))
        axes[2].legend(handles=[matplotlib.patches.Patch(color=c, label=lab) for c, (_, lab) in zip(band, decades, strict=True)]
                       + [matplotlib.patches.Patch(fill=False, ec="black", lw=1.6, label="fastest in the row")],
                       loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=6, fontsize=7, frameon=False)  # fmt: skip
        for ax, key, title in ((axes[0], "worst", "one-step error / tol"), (axes[1], "rollout_worst", "rollout error / tol (20 trajectories)"),
                               (axes[2], None, f"speedup over {BASELINES[baseline]['label']} (top), inference per state (bottom)")):
            for i, n in enumerate(names):
                tols = sorted(PROBLEMS[n].tolerances, reverse=True)
                if key is None or k >= len(tols):
                    continue
                tol = tols[k]
                row = []
                for j, meth in enumerate(methods):
                    path = CACHE / n / f"{meth}_tol{tol:g}.json"
                    if not path.exists():
                        ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, fill=False, ls=":", ec="#bbbbbb"))
                        continue
                    m = json.loads(path.read_text())
                    if "failed" in m:
                        ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, fc="#e8e8e8", hatch="//", ec="#999999"))
                        ax.text(j + 0.47, i + 0.45, "over", ha="center", va="center", fontsize=6.5, color="#555555")
                        continue
                    r = m[key] / tol
                    ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, fc=cmap(norm(r)), ec="white"))
                    ev = m["evals"]
                    txt = f"{ev / 1e3:.1f}k" if ev < 1e5 else f"{ev / 1e6:.2f}M" if ev >= 1e6 else f"{ev / 1e3:.0f}k"
                    ax.text(j + 0.47, i + 0.45, f"{txt}\n{r:.2g}", ha="center", va="center", fontsize=6.5,
                            color="white" if r <= 0.01 or r > 10 else "#222222")  # fmt: skip
                    row.append((r, j))
                if row:  # the row's least error over the tolerance, boxed
                    jb = min(row)[1]
                    ax.add_patch(plt.Rectangle((jb, i), 0.95, 0.9, fill=False, ec="black", lw=1.6))
            if key is not None:  # the error bands, as in the cells
                labels = ["<= 0.01", "0.01-0.1", "0.1-0.5", "0.5-1", "1-2", "2-10", "10-100", "> 100"]
                ax.legend(handles=[matplotlib.patches.Patch(color=cmap(q), label=lab) for q, lab in enumerate(labels)]
                          + [matplotlib.patches.Patch(fc="#e8e8e8", hatch="//", ec="#999999", label="over the time limit"),
                             matplotlib.patches.Patch(fill=False, ec="black", lw=1.6, label="least in the row")],
                          loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=6, fontsize=7, frameon=False,
                          title="worst error / tolerance (blue holds, red exceeds)", title_fontsize=7)  # fmt: skip
            ax.set_xlim(0, len(methods))
            ax.set_ylim(len(names), 0)
            ax.set_xticks(np.arange(len(methods)) + 0.47, [SHORT.get(m, m) for m in methods], fontsize=7, rotation=20, ha="right")
            ax.set_yticks(np.arange(len(names)) + 0.45, [f"{n} ({len(PROBLEMS[n].net.species)})" for n in names], fontsize=7.5)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(title, fontsize=10)
        fig.suptitle(f"Stiff systems, the {'looser' if k == 0 else 'tighter'} tolerance of each problem (0.01 or 0.001, dex or absolute)\n"
                     "top: training solves; bottom: worst error / tolerance; hatched: over the time limit", fontsize=9)  # fmt: skip
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"overview-stiff-{label}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_speedup(baseline: str = BASELINE) -> Path:
    """
    ``output/chem/speedup-stiff.png``: every case as a point, its rollout error over the tolerance against its
    speedup over the named baseline (left of 1 holds the tolerance; up is faster), one colour per method,
    circles for the looser tolerance and squares for the tighter: which methods are both fast and accurate.
    """
    from src.structure.chemfill import ladder

    names = [n for n in ladder(10**6) if (CACHE / n).is_dir()]
    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    handles = []
    for meth, (label, colour, _) in METHODS.items():
        xs, ys, ms = [], [], []
        for n in names:
            for k, tol in enumerate(sorted(PROBLEMS[n].tolerances, reverse=True)):
                path = CACHE / n / f"{meth}_tol{tol:g}.json"
                if not path.exists():
                    continue
                m = json.loads(path.read_text())
                if "failed" in m:
                    continue
                try:
                    ys.append(speedup(m, n, baseline))
                except KeyError:
                    continue
                xs.append(max(m["rollout_worst"] / tol, 1e-4))
                ms.append("o" if k == 0 else "s")
        if not xs:
            continue
        for x, y, mk in zip(xs, ys, ms, strict=True):
            ax.scatter(x, y, s=26, marker=mk, color=colour, alpha=0.8, edgecolors="white", linewidths=0.4)
        handles.append(matplotlib.lines.Line2D([], [], ls="", marker="o", color=colour, label=f"{SHORT.get(meth, meth).replace(chr(10), ' ')} ({len(xs)})"))
    handles += [matplotlib.lines.Line2D([], [], ls="", marker=mk, color="#555555", label=w) for mk, w in (("o", "looser tolerance"), ("s", "tighter tolerance"))]
    ax.axvline(1.0, color="#c0392b", lw=0.8, ls=":")
    ax.axhline(1.0, color="#999999", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rollout error / tolerance (20 trajectories; < 1 holds)")
    ax.set_ylabel(f"speedup over {BASELINES[baseline]['label']}")
    ax.set_title(f"Stiff systems: speed against rollout accuracy. Baseline: {BASELINES[baseline]['integrator']}; "
                 f"{BASELINES[baseline]['device']}. Surrogates: CPU, 1 thread, batched inference", fontsize=8.5, wrap=True)  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "speedup-stiff.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_summary() -> Path:
    lines = ["# Stiff-system bench (errors in dex for log problems, absolute otherwise, over held-out trajectory states; refused states are solved)", "",
             "| problem | tol | method | solves | one-step worst | 99.9% | 99% | % above tol | % refused | rollout worst | train s | infer us | solver us | speedup |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, p in PROBLEMS.items():
        for tol in p.tolerances:
            for method in METHODS:
                path = CACHE / name / f"{method}_tol{tol:g}.json"
                if not path.exists():
                    continue
                m = json.loads(path.read_text())
                if "failed" in m:
                    lines.append(f"| {name} | {tol:g} | {METHODS[method][0]} | {m['failed'][:40]} | | | | | | | | | | |")
                    continue
                lines.append(f"| {name} | {tol:g} | {METHODS[method][0]} | {m['evals']:,} | {m['worst']:.2g} | {m['q999']:.2g} | "
                             f"{m['q99']:.2g} | {m['above_percent']:.2f} | {m['refused_percent']:.2f} | {m.get('rollout_worst', float('nan')):.2g} | {m['train_s']:.1f} | "
                             f"{m['predict_us']:.1f} | {m.get('solve_us', float('nan')):,.0f} | {m.get('speedup', float('nan')):,.0f} |")  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary-chem.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--problems", default="robertson")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tols", default=None, help="default: the problem's tolerances")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot", action="store_true")
    a = parser.parse_args(argv)
    for name in a.problems.split(","):
        for tol in map(float, a.tols.split(",")) if a.tols else PROBLEMS[name].tolerances:
            for method in a.methods.split(","):
                m = run(method, name, tol, force=a.force)
                print(json.dumps({k: v for k, v in m.items() if not k.endswith("by_step")}), flush=True)
        if a.plot:
            plot(name, list(METHODS))
            write_summary()


if __name__ == "__main__":
    main()
