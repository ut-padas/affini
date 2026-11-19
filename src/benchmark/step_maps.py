"""
The ODE bench: surrogates of a system's step map, trained and judged on visited states.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python -m src.benchmark.step_maps \\
        --problems robertson --methods slowtable,chebtt_box --plot

Each problem integrates training and held-out test trajectories once (kept in the store as the problem's
``trajectories.npz``, ``src.benchmark.store``). A surrogate maps a state ``z`` (log10 abundances, and log10 T for the
primordial network) to the increment ``Phi_dt(z) - z``. It is trained from the training states, in
time-major order (every trajectory's first state, then every second state, ...: the order in which a
simulation queries its cells), with access to the oracle; it is judged on every test state, against
the increments the test trajectories already contain, so testing costs no solve.

Metrics per run (one record in the store, seed 1): the error of an answer is the largest
|error| over the components, in dex (log10); ``worst`` is the largest over the answered test states, with
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
``nlu``), since SciPy's per-call overhead flatters a surrogate against a compiled solver. The solver
is the problem's named baseline (``BASELINES``), the integrator its system would be solved with:
Radau for a stiff system, RK45 for a non-stiff one.

Each problem declares its regime (``REGIMES``: stiff, non-stiff, chaotic), and the overviews group
problems by it rather than by the measured ``dynamics``: a method-of-lines family measures non-stiff
on its smallest grids, and the measured class is printed beside each problem instead.
``--plot`` writes ``<problem>-errors.png`` and ``summary-ode.md`` to the suite's figures.
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from src.benchmark.plots import RATIO_COLOURS, RATIO_EDGES, RATIO_NAMES  # noqa: E402
from src.benchmark.store import Store, atomic_write, write_json  # noqa: E402
from src.problems import ode  # noqa: E402
from src.problems.ode import REGIMES, STEP_MAPS, StepMapProblem  # noqa: E402
from src.schemes import chebtt, tabulation  # noqa: E402
from src.schemes.oracle import PointCache  # noqa: E402

# The surrogates train and predict serially, on one CPU core (the solver's pool of processes serves the oracle).
STORE = Store("step_maps", "cpu-1rank")


STEP_TOLERANCES = (0.01, 0.001)  # every step map's, in its units (dex for a log system, else absolute)


FIXED = {"robertson": "robertson", "chemakzo": "chemakzo", "hires": "hires", "primordial": "primordial", "pollu": "pollu",
         "van_der_pol": "van_der_pol", "oregonator": "oregonator", "nelson": "nelson"}  # problem: family


def family(name: str) -> str:
    return FIXED.get(name, name.rsplit("_d", 1)[0])


def ladder(max_d: int, first: tuple[str, ...] = ()) -> list[str]:
    """Problems of the ladder up to ``max_d`` inputs, the families in ``first`` before the rest, each part by input dimension (ties: by name)."""
    names = [n for n in STEP_MAPS if n in FIXED or family(n) != n]
    dims = {n: len(STEP_MAPS[n].system.components) for n in names}
    return sorted((n for n in names if dims[n] <= max_d), key=lambda n: (family(n) not in first, dims[n], n))


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
    "scipy-rk45-cpu1": {
        "label": "SciPy RK45, 1 CPU core, one state at a time",
        "integrator": "scipy.integrate.solve_ivp(method='RK45'): explicit Dormand-Prince 5(4), adaptive steps",
        "settings": "each system's rtol (1e-8) and atol; one call per state and step",
        "device": "CPU, 1 core (OPENBLAS_NUM_THREADS=1)",
        "note": "for non-stiff systems, where an implicit method's linear solves buy nothing; nfev gives a compiled solver's work",
    },
}
BASELINES["scipy-bdf-cpu1"] = {
    "label": "SciPy BDF, 1 CPU core, one state at a time",
    "integrator": "scipy.integrate.solve_ivp(method='BDF'): variable-order BDF (1-5), analytic Jacobian",
    "settings": "the system's rtol and atol (KIDApy networks: 1e-6 and 1e-24); one call per state and step",
    "device": "CPU, 1 core (OPENBLAS_NUM_THREADS=1)",
    "note": "the reference of the KIDApy networks; KIDApy itself runs BDF at rtol 1e-3 in scaled coordinates",
}
BASELINE_OF = {"Radau": "scipy-radau-cpu1", "RK45": "scipy-rk45-cpu1", "BDF": "scipy-bdf-cpu1"}  # a network's integrator: its baseline


def baseline_of(p: StepMapProblem) -> str:
    """The named baseline (``BASELINES``) of the problem's own integrator: what a speedup is over by default."""
    return BASELINE_OF[p.system.method]


def dim(name: str) -> int:
    """A step map's input dimension: its system's components (held inputs included)."""
    return len(STEP_MAPS[name].system.components)


def baseline_cost(name: str, baseline: str | None = None) -> dict:
    """
    One baseline's cost of a step per state on 40 test states (kept in the problem's ``baseline-<baseline>.json``;
    by default the problem's own, ``baseline_of``): wall time in microseconds and the operation counts
    (right-hand sides, Jacobians, LU factorisations), with the baseline's description. Only the system's own
    integrator is measured here (KeyError for another without a record); another baseline's records are written
    by its own runner in the same form.
    """
    p = STEP_MAPS[name]
    baseline = baseline or baseline_of(p)
    path = STORE.data(name, f"baseline-{baseline}.json")
    if path.exists():
        return json.loads(path.read_text())
    if baseline != baseline_of(p):
        raise KeyError(f"no measurement of baseline {baseline!r} for {name}")
    _, test = data(name)
    Z = test[:, :-1].reshape(-1, test.shape[2])
    Z = Z[np.random.default_rng(0).choice(len(Z), min(40, len(Z)), replace=False)]
    cost = p.system.step_cost(10.0**Z if p.system.log else Z, p.dt)
    cost = {**cost, "baseline": baseline, **BASELINES[baseline]}
    write_json(path, cost)
    return cost


def solver_cost(name: str) -> dict:
    """The problem's own baseline's cost (``baseline_cost``)."""
    return baseline_cost(name)


def speedup(m: dict, name: str, baseline: str | None = None) -> float:
    """A run's speedup over a named baseline: the baseline's time per state and step over the surrogate's inference time per state."""
    return baseline_cost(name, baseline)["solve_us"] / max(m["predict_us"], 1e-9)


def data(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) trajectories, (n, steps + 1, d) states in the surrogates' coordinates, integrated once and cached."""
    path = STORE.data(name, "trajectories.npz")
    if path.exists():
        z = np.load(path)
        return z["train"], z["test"]
    p = STEP_MAPS[name]
    seeds = ((p.n_train, 1), (p.n_test, 2))
    start = [p.initial(n, seed) for n, seed in seeds]
    if p.burn > 0:
        start = [p.system.step(Y, p.burn) for Y in start]
    paths = [p.held_path(n, seed, p.steps) if p.held_path else None for n, seed in seeds]
    train, test = (ode.trajectories(p.system, Y, p.dt, p.steps, path) for Y, path in zip(start, paths, strict=True))
    atomic_write(path, lambda tmp: np.savez_compressed(tmp, train=train, test=test))
    return train, test


def visited(train: np.ndarray) -> np.ndarray:
    """The training trajectories' states that start a step, in time-major order (every trajectory's first state, then every second, ...)."""
    return train[:, :-1].transpose(1, 0, 2).reshape(-1, train.shape[2])


def slowtable(p: StepMapProblem, train: np.ndarray, atol: float, seed: int):
    oracle = PointCache(p.step)
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


def chebtt_box(p: StepMapProblem, train: np.ndarray, atol: float, seed: int, oracle: PointCache | None = None, edges: bool = True):
    oracle = oracle or PointCache(p.step)
    lo, hi = state_box(visited(train))
    models = []
    for c in range(train.shape[2]):
        if c >= train.shape[2] - p.system.held:
            models.append(None)  # a held input: increment 0
            continue
        models.append(chebtt.build(lambda X, c=c: (oracle(X) - X)[:, c], lo, hi, atol, seed=seed, max_evals=200_000, edges=edges))
    return BoxTT(models, lo, hi, oracle.evals)


def chebtt_smooth(p: StepMapProblem, train: np.ndarray, atol: float, seed: int):
    """``chebtt_box`` without the edge search: one Chebyshev basis per axis (the control for piecewise axes)."""
    return chebtt_box(p, train, atol, seed, edges=False)


def chebtt_adaptive_box(p: StepMapProblem, train: np.ndarray, atol: float, seed: int):
    """ChebTT-A (``chebtt_adaptive``) per component on the states' bounding box: piecewise axes, and an interface term where one is found."""
    from src.schemes.chebtt import adaptive as chebtt_adaptive

    oracle = PointCache(p.step)
    lo, hi = state_box(visited(train))
    models = [None if c >= train.shape[2] - p.system.held else
              chebtt_adaptive.build(lambda X, c=c: (oracle(X) - X)[:, c], lo, hi, atol, seed=seed, max_evals=200_000)
              for c in range(train.shape[2])]  # fmt: skip
    return BoxTT(models, lo, hi, oracle.evals)


class BoxMethod(BoxTT):
    """A function-suite method (AFINNI or ART, ``src.benchmark.methods.METHODS``) of every component's increment on the states' bounding box."""

    def __init__(self, model, lo, hi, evals, outputs: int, held: int):
        super().__init__(model, lo, hi, evals)
        self.outputs, self.held = outputs, held

    def predict(self, Q):
        Q = np.clip(Q, self.lo, self.hi)
        out = np.asarray(self.models.predict(Q), dtype=np.float64).reshape(len(Q), self.outputs)
        return np.concatenate([out, np.zeros((len(Q), self.held))], axis=1) if self.held else out


def box_method(name: str):
    """A step-map method running the function suite's ``name`` (AFINNI, ART) with the solver as its oracle."""

    def method(p: StepMapProblem, train: np.ndarray, atol: float, seed: int):
        from src import to_host, xp
        from src.benchmark.methods import METHODS as FUNCTION_METHODS

        held = p.system.held
        m = train.shape[2] - held
        lo, hi = state_box(visited(train))
        oracle = PointCache(p.step)

        def f(X):
            Z = np.asarray(to_host(X), dtype=np.float64)
            return xp.asarray((oracle(Z) - Z)[:, :m])

        model = FUNCTION_METHODS[name][2](f, lo, hi, atol, seed)
        return BoxMethod(model, lo, hi, oracle.evals, m, held)

    return method


def chebtt_roll(p: StepMapProblem, train: np.ndarray, atol: float, seed: int, rounds: int = 3):
    """
    ``chebtt_box``, accepted only once the surrogate chained along every training trajectory (projected
    onto the invariants, as it is used) stays within half of ``atol`` of them (a margin for the test
    trajectories, which may pass nearer a sensitive state: Allen-Cahn's unstable u = 0); else retrained
    at the one-step tolerance times the shortfall (at most 8x tighter per round, 0.8 for margin). The training trajectories are known, so the check costs no solve; the retraining reuses
    every cached solve, and ChebTT's degrees grow only logarithmically with the accuracy.
    """
    oracle = PointCache(p.step)
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
        return np.concatenate([out, np.zeros((len(Q), self.held))], axis=1) if self.held else out


def chebtt_metric(p: StepMapProblem, train: np.ndarray, atol: float, seed: int):
    states = visited(train)
    held = p.system.held  # held inputs: their increment is 0
    m = states.shape[1] - held
    lo, hi = states.min(axis=0), states.max(axis=0)
    pad = 0.02 * np.maximum(hi - lo, 1e-3)
    lo, hi = lo - pad, hi + pad
    oracle = PointCache(p.step)
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
           "afinni": ("AFINNI", "#6d3b0f", box_method("afinni")),
           "art": ("ART, affine leaves", "#1f5fa8", box_method("art")),
           "art_quad": ("ART, quadratic leaves", "#2e7d32", box_method("art_quad"))}  # fmt: skip


def invariants_of(p: StepMapProblem, train: np.ndarray) -> np.ndarray:
    """
    (m, d) the system's linear invariants W (W F(y) = 0 for every y: element and charge totals of a
    network, total mass), from the null space of the right-hand side at up to 600 training states (the
    held temperature excluded), rows orthonormal; empty when there are none. In the system's own
    variables (abundances, not their log10).
    """
    Z = visited(train)
    Z = Z[np.random.default_rng(0).choice(len(Z), min(600, len(Z)), replace=False)]
    free = list(range(Z.shape[1] - p.system.held))
    Y = 10.0**Z if p.system.log else Z
    F = np.array([np.asarray(p.system.rhs(y))[free] for y in Y])
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

    def __init__(self, model, p: StepMapProblem, W: np.ndarray):
        self.model, self.p, self.W = model, p, W
        self.evals = getattr(model, "evals", 0)

    def refuses(self, Z):
        return self.model.refuses(Z)

    def predict(self, Z):
        inc = self.model.predict(Z)
        log = self.p.system.log
        y, yh = (10.0**Z, 10.0 ** (Z + inc)) if log else (Z, Z + inc)
        D2 = yh * yh if log else np.ones_like(yh)
        r = (y - yh) @ self.W.T  # (N, m) the invariants' defect
        M = np.einsum("ij,nj,kj->nik", self.W, D2, self.W)
        ys = yh + D2 * np.einsum("nm,mj->nj", np.linalg.solve(M, r[:, :, None])[:, :, 0], self.W)
        return (np.log10(np.maximum(ys, ode.FLOOR)) if log else ys) - Z


def run(method: str, name: str, tol: float, seed: int = 1, force: bool = False) -> dict:
    known = STORE.read(name, dim(name), method, tol, seed)
    if known is not None and not force:
        return known
    p = STEP_MAPS[name]
    train, test = data(name)
    states = train[:, :-1].transpose(1, 0, 2).reshape(-1, train.shape[2])  # time-major
    Zt, Gt = test[:, :-1].reshape(-1, test.shape[2]), (test[:, 1:] - test[:, :-1]).reshape(-1, test.shape[2])
    t0 = time.perf_counter()
    try:
        model = METHODS[method][2](p, train, tol, seed)
    except RuntimeError as err:
        m = {"method": method, "problem": name, "d": dim(name), "tol": tol, "seed": seed, "failed": str(err)[:300]}
        STORE.write(m)
        return m
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(Zt)
    predict_us = 1e6 * (time.perf_counter() - t0) / len(Zt)
    refused = np.asarray(model.refuses(Zt), dtype=bool)
    mask = p.scored(test[:, 1:].reshape(-1, test.shape[2]))
    e = np.max(np.where(mask, np.abs(pred - Gt), 0.0), axis=1)
    ea = e[~refused] if (~refused).any() else np.zeros(1)
    m = {"method": method, "problem": name, "d": dim(name), "tol": tol, "seed": seed, "evals": int(model.evals), "train_states": len(states),
         "worst": float(ea.max()), "worst_ratio": float(ea.max()) / tol, "q999": float(np.quantile(ea, 0.999)), "q99": float(np.quantile(ea, 0.99)),
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
    m["speedup"] = m["solve_us"] / max(predict_us, 1e-9)  # over the problem's baseline: its time per state and step over the surrogate's
    m["inference"] = {"device": "cpu", "threads": int(os.environ.get("OPENBLAS_NUM_THREADS", "0") or 0), "batch": len(Zt),
                      "baseline": baseline_of(p)}  # fmt: skip
    STORE.write(m)
    return m


ROLLOUTS = 20  # test trajectories chained through each surrogate


def survived(p: StepMapProblem, model, test: np.ndarray, n: int = 200) -> float:
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


def rollout(p: StepMapProblem, model, true: np.ndarray) -> dict:
    """
    Chain the surrogate from each true initial state for every step of the trajectory, the solver
    stepping the states it refuses (as a simulation falls back), and compare with the solver's
    trajectory: the error a one-zone run accumulates, which the one-step error alone does not show.
    """
    Z = true[:, 0].copy()
    err, solved = [], []
    h = p.system.held
    for k in range(true.shape[1] - 1):
        if h:
            Z[:, -h:] = true[:, k, -h:]  # held inputs (an environment) come from the simulation, not the surrogate
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
    path = STORE.data(name, "dynamics.json")
    if path.exists():
        return json.loads(path.read_text())
    p = STEP_MAPS[name]
    _, test = data(name)
    true = test[:ROLLOUTS]  # the rollouts' own trajectories, so their errors and A_n refer to the same states
    n_steps, d = true.shape[1] - 1, true.shape[2]
    free = list(range(d - p.system.held))
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
            y = 10.0 ** traj[k] if p.system.log else traj[k]
            lam = np.abs(np.real(np.linalg.eigvals(np.asarray(p.system.jac(y))[np.ix_(free, free)])))
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
    write_json(path, out)
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
    tols = sorted(STEP_TOLERANCES, reverse=True)
    handles = []
    for method in methods:
        solves = []
        for tol, ls in zip(tols, ("-", "--"), strict=False):
            m = STORE.read(name, dim(name), method, tol, 1)
            if m is None:
                continue
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
    handles += [matplotlib.lines.Line2D([], [], color="#555555", ls=ls, lw=1.3, label=f"tol {t:g} {STEP_MAPS[name].units}")
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
    p = STEP_MAPS[name]
    units = "log10 (dex)" if p.system.log else "absolute units"
    try:
        cost = solver_cost(name)
        solver = f"; the solver: {cost['solve_us']:,.0f} us per state and step ({cost['nfev']:.0f} RHS, {cost['nlu']:.0f} LU)"
    except Exception:  # noqa: BLE001 (the cost is a caption only)
        solver = ""
    fig.suptitle(f"{name}: surrogates of the step map (dt = {p.dt:.3g}), on held-out trajectories; errors in {units}{solver}", fontsize=10)
    path = STORE.figures / f"{name}-errors.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_ladder() -> Path:
    """
    ``output/ode/ladder.png``: per family (columns; the fixed-size test problems together), against the
    input dimension, the one-step worst error, the part of it that survives the next exact step and the
    rollout worst error (all over the tolerance), the training solves, and the speedup (the solver's time per state and step over the surrogate's). One
    colour per method, solid for the looser tolerance and dashed for the tighter; an x at the top of
    the error panels marks a case over the time limit or the evaluation budget.
    """

    names = ladder(10**6)
    fams = [f for f in dict.fromkeys(family(n) for n in names) if f not in FIXED] + ["test set"]
    rows = (("worst", "one-step worst / tol"), ("survived_worst", "one-step error surviving the next step / tol"),
            ("rollout_worst", "rollout worst / tol"), ("evals", "training solves"), ("speedup", "solver time / inference time"))  # fmt: skip
    fig, axes = plt.subplots(len(rows), len(fams), figsize=(4.2 * len(fams), 13.5), squeeze=False, layout="constrained")
    for j, fam in enumerate(fams):
        members = [n for n in names if (family(n) == fam) or (fam == "test set" and n in FIXED)]
        for method, (_, colour, _) in METHODS.items():
            for k, ls in enumerate(("-", "--")):
                pts, fails = {key: [] for key, _ in rows}, []
                for n in members:
                    p = STEP_MAPS[n]
                    if k >= len(STEP_TOLERANCES):
                        continue
                    tol = sorted(STEP_TOLERANCES, reverse=True)[k]
                    m = STORE.read(n, dim(n), method, tol, 1)
                    if m is None:
                        continue
                    d = len(p.system.components)
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
        for i, (_, ylabel) in enumerate(rows):
            ax = axes[i, j]
            ax.set_yscale("log")
            ax.set_xscale("log")
            if i < 3:
                ax.axhline(1.0, color="#c0392b", lw=0.8, ls=":")
            ax.set_ylabel(ylabel if j == 0 else "")
            ax.set_xlabel("input dimension" if i == len(rows) - 1 else "")
            dims = sorted({len(STEP_MAPS[n].system.components) for n in members})
            if dims:
                ax.set_xticks(dims, [str(d) for d in dims])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        title = fam if fam != "test set" else "Robertson (3), Akzo Nobel (5), HIRES (8), POLLU (20)"
        axes[0, j].set_title(title, fontsize=10)
    shown = [m for m in METHODS if any(STORE.exists(n, dim(n), m, t, 1) for n in names for t in STEP_TOLERANCES)]
    handles = [matplotlib.lines.Line2D([], [], color=METHODS[m][1], lw=2, marker="o", ms=4, label=SHORT.get(m, m).replace(chr(10), " ")) for m in shown]
    handles += [matplotlib.lines.Line2D([], [], color="#555555", ls=ls, label=f"{w} tolerance") for w, ls in (("looser", "-"), ("tighter", "--"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(6, len(handles)), fontsize=8, frameon=False)
    fig.suptitle("ODE ladder: surrogates of the step map on held-out trajectories (x at the top: over the time limit or budget)",
                 fontsize=11)  # fmt: skip
    path = STORE.figures / "ladder.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


SHORT = {"slowtable": "SlowTable", "chebtt_smooth": "ChebTT\nno edges", "chebtt_box": "ChebTT", "chebtt_adaptive": "ChebTT-A",
         "chebtt_metric": "ChebTT-M", "chebtt_roll": "ChebTT\nrollout", "afinni": "AFINNI", "art": "ART\naffine", "art_quad": "ART\nquad"}  # fmt: skip


def _baseline_label(names: list[str], baseline: str | None) -> str:
    """What the speedups of these problems are over: the named baseline, or each problem's own when they differ."""
    own = {baseline or baseline_of(STEP_MAPS[n]) for n in names}
    return BASELINES[own.pop()]["label"] if len(own) == 1 else "each problem's own baseline (" + ", ".join(sorted(own)) + ")"


def _row_label(n: str) -> str:
    """A problem's row: its name, then its input dimension, its system's parameters and its measured dynamics class (when diagnosed)."""
    system, path = STEP_MAPS[n].system, STORE.data(n, "dynamics.json")
    parts = [str(len(system.components)), system.params, json.loads(path.read_text())["kind"] if path.exists() else ""]
    return f"{n} ({'; '.join(filter(None, parts))})"


def _regime_problems() -> dict[str, list[str]]:
    """Per regime (``REGIMES`` order), the ladder's problems with records, by input dimension; empty regimes left out."""

    names = [n for n in ladder(10**6) if STORE.runs(problem=n)]
    return {r: [n for n in names if STEP_MAPS[n].regime == r] for r in REGIMES if any(STEP_MAPS[n].regime == r for n in names)}


def plot_overview(baseline: str | None = None) -> list[Path]:
    """
    ``output/ode/overview-<regime>-{loose,tight}.png``, one pair per regime: per problem (rows, by input
    dimension) and method (columns), the worst one-step error and the worst rollout error over the
    tolerance (blue within, red beyond), with the training solves, and the speedup over ``baseline``
    (default: each problem's own); hatched: over the time limit or the evaluation budget.
    """
    paths = []
    for regime, names in _regime_problems().items():
        paths += _overview(regime, names, baseline)
    return paths


def _overview(regime: str, names: list[str], baseline: str | None) -> list[Path]:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    methods = [m for m in METHODS if any(STORE.exists(n, dim(n), m, t, 1) for n in names for t in STEP_TOLERANCES)]
    over = _baseline_label(names, baseline)
    bounds = [0, *RATIO_EDGES, 1e300]
    cmap = ListedColormap(list(RATIO_COLOURS))
    norm = BoundaryNorm(bounds, cmap.N)
    paths = []
    for k, label in ((0, "loose"), (1, "tight")):
        fig, axes = plt.subplots(3, 1, figsize=(max(0.82 * len(methods) + 2.6, 9.5), 3 * (0.34 * len(names) + 1.0) + 1.4), layout="constrained")
        # third panel: speedup over the named baseline, in decade bands of a perceptually uniform,
        # colour-blind-safe scale (viridis), the row's fastest boxed
        decades = [(1e1, "< 10x"), (1e2, "10-100x"), (1e3, "100-1,000x"), (1e4, "1,000-10,000x"), (np.inf, ">= 10,000x")]
        band = [plt.get_cmap("viridis")(v) for v in (0.0, 0.3, 0.55, 0.75, 0.97)]
        for i, n in enumerate(names):
            tols = sorted(STEP_TOLERANCES, reverse=True)
            if k >= len(tols):
                continue
            row = []
            for j, meth in enumerate(methods):
                m = STORE.read(n, dim(n), meth, tols[k], 1)
                if m is None:
                    continue
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
        speed_key = [matplotlib.patches.Patch(color=c, label=lab) for c, (_, lab) in zip(band, decades, strict=True)]
        speed_key += [matplotlib.patches.Patch(fill=False, ec="black", lw=1.6, label="fastest in the row")]
        for ax, key, title in ((axes[0], "worst", "one-step error / tol"), (axes[1], "rollout_worst", "rollout error / tol (20 trajectories)"),
                               (axes[2], None, f"speedup over {over} (top), inference per state (bottom)")):
            for i, n in enumerate(names):
                tols = sorted(STEP_TOLERANCES, reverse=True)
                if key is None or k >= len(tols):
                    continue
                tol = tols[k]
                row = []
                for j, meth in enumerate(methods):
                    m = STORE.read(n, dim(n), meth, tol, 1)
                    if m is None:
                        ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, fill=False, ls=":", ec="#bbbbbb"))
                        continue
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
            ax.set_xlim(0, len(methods))
            ax.set_ylim(len(names), 0)
            ax.set_xticks(np.arange(len(methods)) + 0.47, [SHORT.get(m, m) for m in methods], fontsize=7, rotation=20, ha="right")
            ax.set_yticks(np.arange(len(names)) + 0.45, [_row_label(n) for n in names], fontsize=7.5)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(title, fontsize=10)
        # one key for the two error tables, above them; the speedup key below its table
        labels = list(RATIO_NAMES)
        error_key = [matplotlib.patches.Patch(color=cmap(q), label=lab) for q, lab in enumerate(labels)]
        error_key += [matplotlib.patches.Patch(fc="#e8e8e8", hatch="//", ec="#999999", label="over the time limit"),
                      matplotlib.patches.Patch(fill=False, ec="black", lw=1.6, label="least in the row")]
        # the figure's heading is this key's title: constrained layout does not stack a suptitle and an outside legend
        fig.legend(handles=error_key, loc="outside upper center", ncol=10, fontsize=7, frameon=False, handlelength=1.4, columnspacing=1.0,
                   title=f"{REGIMES[regime]}, the {'looser' if k == 0 else 'tighter'} tolerance of each problem (0.01 or 0.001, dex or absolute).\n"
                         "Error cells: training solves above the worst error / tolerance (blue holds, red exceeds)", title_fontsize=8.5)
        fig.legend(handles=speed_key, loc="outside lower center", ncol=6, fontsize=7, frameon=False, handlelength=1.4,
                   title=f"speedup over {over}", title_fontsize=7.5)  # fmt: skip
        path = STORE.figures / f"overview-{regime}-{label}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_speedup(baseline: str | None = None) -> list[Path]:
    """
    ``output/ode/speedup-<regime>.png``, one per regime: every case as a point, its rollout error over the
    tolerance against its speedup over ``baseline`` (default: each problem's own; left of 1 holds the
    tolerance; up is faster), one colour per method, circles for the looser tolerance and squares for the
    tighter: which methods are both fast and accurate.
    """
    return [_speedup(regime, names, baseline) for regime, names in _regime_problems().items()]


def _speedup(regime: str, names: list[str], baseline: str | None) -> Path:
    own = {baseline or baseline_of(STEP_MAPS[n]) for n in names}
    fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
    handles = []
    for meth, (_, colour, _) in METHODS.items():
        xs, ys, ms = [], [], []
        for n in names:
            for k, tol in enumerate(sorted(STEP_TOLERANCES, reverse=True)):
                m = STORE.read(n, dim(n), meth, tol, 1)
                if m is None:
                    continue
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
    ax.set_ylabel(f"speedup over {_baseline_label(names, baseline)}")
    base = "; ".join(f"{BASELINES[b]['integrator']}, {BASELINES[b]['device']}" for b in sorted(own))
    ax.set_title(f"{REGIMES[regime]}: speed against rollout accuracy. Baseline: {base}. Surrogates: CPU, 1 thread, batched inference",
                 fontsize=8.5, wrap=True)  # fmt: skip
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=8, frameon=False)
    path = STORE.figures / f"speedup-{regime}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_summary() -> Path:
    lines = ["# ODE bench (errors in dex for log problems, absolute otherwise, over held-out trajectory states; refused states are solved)", "",
             "| problem | tol | method | solves | one-step worst | 99.9% | 99% | % above tol | % refused | rollout worst | train s | infer us | solver us | speedup |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name in STEP_MAPS:
        for tol in STEP_TOLERANCES:
            for method in METHODS:
                m = STORE.read(name, dim(name), method, tol, 1)
                if m is None:
                    continue
                if "failed" in m:
                    lines.append(f"| {name} | {tol:g} | {METHODS[method][0]} | {m['failed'][:40]} | | | | | | | | | | |")
                    continue
                lines.append(f"| {name} | {tol:g} | {METHODS[method][0]} | {m['evals']:,} | {m['worst']:.2g} | {m['q999']:.2g} | "
                             f"{m['q99']:.2g} | {m['above_percent']:.2f} | {m['refused_percent']:.2f} | {m.get('rollout_worst', float('nan')):.2g} | {m['train_s']:.1f} | "
                             f"{m['predict_us']:.1f} | {m.get('solve_us', float('nan')):,.0f} | {m.get('speedup', float('nan')):,.0f} |")  # fmt: skip
    path = STORE.figures / "summary-ode.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def redraw(name: str) -> None:
    """Every plot a new record of ``name`` changes: its error plot, the summary, the ladder, and every regime's overview and speed plot."""
    plot(name, list(METHODS))
    write_summary()
    plot_ladder()
    plot_overview()
    plot_speedup()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--problems", default="robertson")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--tols", default=None, help="default: the problem's tolerances")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--redraw", action="store_true", help="only redraw every plot a record of these problems appears in (no runs)")
    a = parser.parse_args(argv)
    if a.redraw:
        for name in a.problems.split(","):
            redraw(name)
        return
    for name in a.problems.split(","):
        for tol in map(float, a.tols.split(",")) if a.tols else STEP_TOLERANCES:
            for method in a.methods.split(","):
                m = run(method, name, tol, force=a.force)
                print(json.dumps({k: v for k, v in m.items() if not k.endswith("by_step")}), flush=True)
        if a.plot:
            plot(name, list(METHODS))
            write_summary()


if __name__ == "__main__":
    main()
