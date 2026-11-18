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
records how much the solver's own dynamics grow an early error, to tell the two causes of drift apart.
``--plot`` writes ``output/chem/<problem>-errors.png`` and ``summary-chem.md``.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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


def solver_cost(name: str) -> dict:
    """The solver's cost of one step per state (``Network.step_cost``) on 40 test states, cached."""
    path = CACHE / f"{name}-cost.json"
    if path.exists():
        return json.loads(path.read_text())
    p = PROBLEMS[name]
    _, test = data(name)
    Z = test[:, :-1].reshape(-1, test.shape[2])
    Z = Z[np.random.default_rng(0).choice(len(Z), min(40, len(Z)), replace=False)]
    cost = p.net.step_cost(10.0**Z if p.net.log else Z, p.dt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cost))
    return cost


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


def slowtable(p: ChemProblem, states: np.ndarray, atol: float, seed: int):
    oracle = CachedOracle(p.step)
    t = tabulation.build(oracle, states, atol, seed=seed)
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


def chebtt_box(p: ChemProblem, states: np.ndarray, atol: float, seed: int):
    oracle = CachedOracle(p.step)
    lo, hi = states.min(axis=0), states.max(axis=0)
    pad = 0.02 * np.maximum(hi - lo, 1e-3)
    lo, hi = lo - pad, hi + pad
    models = []
    for c in range(states.shape[1]):
        if p.net.species[c] == "T":
            models.append(None)  # the held temperature: increment 0
            continue
        flat = hi - lo <= 2e-3  # a coordinate the data never varies (kept fixed in the box)
        lo_c, hi_c = np.where(flat, lo - 1e-3, lo), np.where(flat, hi + 1e-3, hi)
        models.append(chebtt.build(lambda X, c=c: (oracle(X) - X)[:, c], lo_c, hi_c, atol, seed=seed, max_evals=200_000))
    return BoxTT(models, lo, hi, oracle.evals)


class BoxVectorTT(BoxTT):
    """``chebtt.build_vector`` (rotated per output group where that pays) on the padded bounding box of the training states."""

    def predict(self, Q):
        Q = np.clip(Q, self.lo, self.hi)
        out = self.models.predict(Q)
        return np.concatenate([out, np.zeros((len(Q), 1))], axis=1) if self.held else out


def chebtt_metric(p: ChemProblem, states: np.ndarray, atol: float, seed: int):
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
           "chebtt_box": ("ChebTT per species on the states' bounding box", "#e08a00", chebtt_box),
           "chebtt_metric": ("ChebTT per species in the pullback-metric coordinates", "#1b7f5b", chebtt_metric)}  # fmt: skip


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
        model = METHODS[method][2](p, states, tol, seed)
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
    m["speedup"] = m["solve_us"] / max(predict_us, 1e-9)  # the solver's time per state and step over the surrogate's
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m))
    return m


ROLLOUTS = 20  # test trajectories chained through each surrogate


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


def amplification(name: str) -> np.ndarray:
    """
    (steps,) the largest singular value of ``J_k ... J_1`` along the first test trajectories, J the
    solver's step-map Jacobian in log10 coordinates (central differences): how much the dynamics alone
    can grow an error made at the first step after k steps, the same for every surrogate. Cached.
    """
    path = CACHE / f"{name}-amplification.npz"
    if path.exists():
        return np.load(path)["growth"]
    p = PROBLEMS[name]
    _, test = data(name)
    true = test[: min(ROLLOUTS, 5)]
    d = true.shape[2]
    free = [i for i in range(d) if p.net.species[i] != "T"]
    h = 1e-5
    growth = np.zeros((len(true), true.shape[1] - 1))
    for t, traj in enumerate(true):
        P = np.eye(len(free))
        for k in range(traj.shape[0] - 1):
            E = np.zeros((2 * len(free), d))
            for j, i in enumerate(free):
                E[2 * j, i], E[2 * j + 1, i] = h, -h
            out = p.step(traj[k][None] + E)[:, free]
            J = (out[0::2] - out[1::2]).T / (2 * h)
            P = J @ P
            growth[t, k] = np.linalg.norm(P, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, growth=growth.max(axis=0))
    return growth.max(axis=0)


def plot(name: str, methods: list[str]) -> Path:
    """
    Four panels over the steps of the test trajectories: the one-step error (a surrogate step from the
    true state against the solver's step: how well the visited states are covered), the rollout error
    (the surrogate chained from the initial state, the solver stepping what it refuses, against the
    solver's trajectory), the share of states the surrogate refuses, and the growth the dynamics alone
    allow an early error (the same for every surrogate).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    (a, b), (c, g) = axes
    for method in methods:
        for tol, ls in zip(PROBLEMS[name].tolerances, ("-", "--"), strict=False):
            path = CACHE / name / f"{method}_tol{tol:g}.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            if "failed" in m:
                continue
            speed = f", {m['speedup']:,.0f}x the solver's speed" if "speedup" in m else ""
            label = f"{METHODS[method][0]}, tol {tol:g} {PROBLEMS[name].units} ({m['evals']:,} training solves{speed})"
            colour = METHODS[method][1]
            a.semilogy(np.maximum(m["error_by_step"], 1e-12) / tol, ls, color=colour, label=label)
            if "rollout_error_by_step" in m:
                b.semilogy(np.maximum(m["rollout_error_by_step"], 1e-12) / tol, ls, color=colour)
            c.plot(100 * np.array(m["refused_by_step"]), ls, color=colour)
    try:
        g.semilogy(amplification(name), color="#444444")
    except Exception as err:  # noqa: BLE001 (the panel is optional; a failed solve must not lose the others)
        g.text(0.5, 0.5, f"not computed: {err}", transform=g.transAxes, ha="center", fontsize=8)
    for ax in (a, b):
        ax.axhline(1.0, color="#c0392b", lw=0.8, ls=":")
    g.axhline(1.0, color="#999999", lw=0.8, ls=":")
    a.set_title("one-step error: a surrogate step from the true state vs the solver's step", fontsize=9)
    b.set_title("rollout error: the surrogate chained from the initial state vs the solver's trajectory", fontsize=9)
    c.set_title("states the surrogate refuses (the solver steps them)", fontsize=9)
    g.set_title("error growth the dynamics alone allow: largest singular value of J_k ... J_1", fontsize=9)
    a.set_ylabel("worst |error| / tolerance (answered states)")
    b.set_ylabel("worst |error| / tolerance over 20 rollouts")
    c.set_ylabel("% of test states refused")
    g.set_ylabel("amplification of an error made at step 1")
    for ax in axes.ravel():
        ax.set_xlabel("step of the test trajectories")
    a.legend(fontsize=7, frameon=False)
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
    input dimension, the one-step worst error and the rollout worst error (both over the tolerance), the
    training solves, and the speedup (the solver's time per state and step over the surrogate's). One
    colour per method, solid for the looser tolerance and dashed for the tighter; an x at the top of
    the error panels marks a case over the time limit or the evaluation budget.
    """
    from src.structure.chemfill import FIXED, family, ladder

    names = ladder(10**6)
    fams = [f for f in dict.fromkeys(family(n) for n in names) if f not in FIXED] + ["test set"]
    rows = (("worst", "one-step worst / tol"), ("rollout_worst", "rollout worst / tol"), ("evals", "training solves"),
            ("speedup", "solver time / inference time"))  # fmt: skip
    fig, axes = plt.subplots(len(rows), len(fams), figsize=(4.2 * len(fams), 10.5), squeeze=False, constrained_layout=True)
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
                            pts[key].append((d, v / tol if key in ("worst", "rollout_worst") else v))
                for i, (key, _) in enumerate(rows):
                    if pts[key]:
                        x, y = zip(*sorted(pts[key]), strict=True)
                        axes[i, j].plot(x, y, ls, marker="o", ms=3.5, color=colour,
                                        label=f"{label}, tol {'loose' if k == 0 else 'tight'}" if i == 0 and j == 0 else None)  # fmt: skip
                for i in (0, 1):
                    for d in fails:
                        axes[i, j].plot(d, 30.0, "x", color=colour, ms=7)
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            ax.set_yscale("log")
            ax.set_xscale("log")
            if i < 2:
                ax.axhline(1.0, color="#c0392b", lw=0.8, ls=":")
            ax.set_ylabel(ylabel if j == 0 else "")
            ax.set_xlabel("input dimension" if i == len(rows) - 1 else "")
            dims = sorted({len(PROBLEMS[n].net.species) for n in members})
            if dims:
                ax.set_xticks(dims, [str(d) for d in dims])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        title = fam if fam != "test set" else "Robertson (3), Akzo Nobel (5), HIRES (8), POLLU (20)"
        axes[0, j].set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Stiff-system ladder: surrogates of the step map on held-out trajectories (x at the top: over the time limit or budget)",
                 fontsize=11)  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "ladder.png"
    fig.savefig(path, dpi=120)
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
