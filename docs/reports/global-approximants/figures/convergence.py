"""The convergence section of the schemes' companion: how each scheme's evaluations and error grow as the tolerance shrinks, beside AFINNI's and ART's.

Usage, from the repository root (``make report-convergence``, about 70 minutes on a laptop's CPU):

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python \\
        docs/reports/global-approximants/figures/convergence.py

Runs the four schemes through ``src.schemes`` at tolerances a decade apart on four problems: Rosenbrock's
valley along a chain (d = 2, 3, 5, the problem of the adaptive companion's convergence study), the feature
example's ridge lifted to d = 2, 3, 5, the feature example F itself, and the hub expansion's two 10-D problems.
AFINNI and ART run on the ridge and on F at tolerances that halve, through ``baseline_on``: on the ridge with
the settings of the adaptive companion's study, on F with this companion's (section 7). Their runs on
Rosenbrock are that study's, read from its convergence.json. A scheme that fails at one tolerance is not run at
tighter ones. Every model is measured on the benchmark's test sets. The script stops unless the adaptive study
ran the same function on the same box, and unless ChebTT-M's rotated box has the half-width along the ridge
that the degree prediction assumes. Writes convergence.json and the convergence-*.svg panels.

Every case records its training time (the build, the oracle's evaluations included), the part of it spent in
the oracle, its peak memory, its inference time per query on the test set and its saved model's size; the
methods run on one CPU core, and Robertson's oracle solves on ``SURROGATE_ODE_WORKERS`` processes. Every case
(one method at one tolerance on one problem) is a record of ``src.infra.records`` under
results/convergence/global-approximants/, written as soon as it finishes, so a run that stops resumes at the
first unfinished case. A record is reused unless its method's results changed after it was written
(``RESULTS_CHANGED`` in ``src.benchmark.table``); a change to the problem, its test points or the case's
settings makes a new case.
"""

import json
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # before NumPy loads BLAS: the hub expansion's Gram products depend on it, and the times are one core's

from pathlib import Path  # noqa: E402

import make_figures as gm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import RESULTS_DIR  # noqa: E402
from src.benchmark.table import RESULTS_CHANGED  # noqa: E402
from src.config import ARTSettings  # noqa: E402
from src.infra.records import (  # noqa: E402
    Probe,
    RecordStore,
    data_hash,
    error_stats,
    per_query_us,
    provenance,
    saved_bytes,
    timed_oracle,
)
from src.problems import FUNCTIONS  # noqa: E402
from src.schemes import base  # noqa: E402

OUT = Path(__file__).resolve().parent
ADAPTIVE = OUT.parents[1] / "adaptive-surrogates" / "figures" / "convergence.json"
SCHEMES = gm.SCHEMES
DECADES = {"rosen": [10.0**-k for k in range(0, 11, 2)], "ridge": [10.0**-k for k in range(1, 11)],
           "F": [10.0**-k for k in range(1, 7)], "hub": [10.0**-k for k in range(1, 7)]}  # fmt: skip
DIMS = (2, 3, 5)
# AFINNI's and ART's tolerances on the ridge: the loosest, and the halvings of each (AFINNI's, ART's), ending
# where a run takes about two minutes on one laptop CPU or AFINNI's sample would exceed the adaptive study's
# largest. ART needs few leaves on a ridge, so it goes further.
RIDGE_HALVINGS = {2: (0.1, 14, 14), 3: (0.1, 14, 14), 5: (0.1, 7, 12)}
POINTS_PER_LEAF = MAX_SAMPLE = None  # the adaptive study's, read from its convergence.json
STORE = RecordStore(RESULTS_DIR / "convergence" / "global-approximants")
F_HALVINGS = (0.1, 4)
HUB_D = gm.HUB_D
# The vector problems: (d, the schemes' tolerances), a decade apart; and the stiff step map's tolerances, in dex
# (the schemes', a decade apart; AFINNI's and ART's, halving), ending where a run takes about two minutes
# with 12 ODE workers.
VECTOR = {"shared_latent": (5, [10.0**-k for k in range(1, 9)]), "lorenz96_state": (5, [10.0**-k for k in range(1, 9)])}
STIFF = ([10.0**-k for k in range(1, 7)], [0.1 / 2**k for k in range(7)], [0.1 / 2**k for k in range(10)])


def rosen_box(d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.r_[gm.R_LO[0], np.full(d - 1, gm.R_LO[1])], np.r_[gm.R_HI[0], np.full(d - 1, gm.R_HI[1])]


def test_points(lo, hi) -> np.ndarray:
    """The benchmark's 2-D grid, or 100,000 fixed uniform points above it."""
    d = len(lo)
    return gm.grid(lo, hi, 300) if d == 2 else np.random.default_rng(20_261_119).uniform(lo, hi, (100_000, d))


def structure(m) -> dict:
    """The ranks and degrees of a ChebTT-like model (ChebTT-M: of its first group, in its coordinates); the hub's groups."""
    if hasattr(m, "describe"):
        return {"groups": [{"r": int(g[1]), "degree": int(g[2]), "weak": bool(g[3]), "terms": int(g[4])} for g in m.describe()]}
    if hasattr(m, "groups"):  # ChebTT-M, or ChebTT on several outputs: the first group's first output, and over all
        _, Q, models = m.groups[0]
        every = [tt for _, _, ms in m.groups for tt in ms]
        return {"rotated": Q is not None, **structure(models[0]), "max_rank": max(max(tt.ranks, default=1) for tt in every),
                "rotated_outputs": sum(len(cols) for cols, Qg, _ in m.groups if Qg is not None)}  # fmt: skip
    if hasattr(m, "models"):  # ChebTT-A on several outputs
        return {"max_rank": max(max(a.tt.ranks, default=1) for a in m.models),
                "interfaces": sum(a.interface is not None for a in m.models)}
    tt = getattr(m, "tt", m)
    out = {"ranks": list(map(int, tt.ranks)), "degrees": list(map(int, tt.degrees))}
    if getattr(m, "interface", None) is not None:  # ChebTT-A's term S: a profile h(phi), or jump fields of a degree
        i = m.interface
        out["interface"] = {"kind": i.kind, "profile": i.profile is not None, "fields_degree": int(i.degree)}
    return out


def scheme_ladder(name: str, f, lo, hi, tols, X, truth, extra=None, outputs: int = 0) -> list[dict]:
    """
    The scheme at each tolerance until one fails; per rung its evaluations, worst error (over the points and the
    outputs) in units of tol, structure. ``outputs``: f is vector-valued, and the scheme's vector build runs.
    """

    def case(tol: float) -> dict:
        g, spent = gm.counted(f)
        g, oracle = timed_oracle(g)
        failed = None
        with Probe() as probe:
            try:
                m = gm.scheme_on(name, g, lo, hi, tol, outputs)
            except RuntimeError as e:
                failed = str(e).split(".")[0]
        cost = {"train_s": gm.r3(probe.seconds), "oracle_s": gm.r3(oracle["s"]), "oracle_calls": oracle["calls"],
                "peak_host_mb": probe.peak_mb, "peak_device_mb": probe.device_mb}  # fmt: skip
        if failed is not None:
            return {"tol": tol, "failed": failed, "evals": int(spent[0]), **cost}
        P, us = per_query_us(m.predict, X)
        E = np.abs(np.asarray(P, dtype=np.float64).reshape(truth.shape) - truth)
        row = {"tol": tol, "evals": int(m.evals), **error_stats(E.reshape(len(X), -1).max(axis=1) / tol),
               "worst": gm.r3(float(np.max(E)) / tol), **cost, "predict_us": gm.r3(us), "model_bytes": saved_bytes(base.save, m),
               "n_test": len(X), **structure(m)}  # fmt: skip
        if extra is not None:
            extra(m, row)
        return row

    rows = []
    for tol in tols:
        key = {"kind": "scheme", "method": name, "data": data_hash(lo, hi, X, truth), "tol": repr(tol),
               "settings": {"outputs": outputs, "extra": getattr(extra, "__name__", None)}}  # fmt: skip
        rows.append(STORE.get(key, case, tol, changed=RESULTS_CHANGED, method=name))
        if "failed" in rows[-1]:
            break
        print(name, len(lo), rows[-1], flush=True)
    return rows


def baseline_ladder(name: str, f, lo, hi, tols, X, truth, size, **kw) -> list[dict]:
    """
    AFINNI or ART at each tolerance, from ``size(rows)`` initial points (None: ``baseline_on``'s), until a sample
    would exceed the adaptive study's largest: leaves, evaluations, worst and 99th-percentile error inside, share flagged.
    """

    def case(tol: float, n0: int | None) -> dict:
        g, oracle = timed_oracle(f)
        with Probe() as probe:
            m = gm.baseline_on(name, lo, hi, tol, f=g, n0=n0, **kw)
        _, us = per_query_us(m.predict, X)  # the plain inference; the errors below come from the flagged one
        (P, outside), flagged_us = per_query_us(m.predict_flagged, X)
        e = np.abs(np.asarray(P).reshape(truth.shape) - truth).reshape(len(X), -1).max(axis=1)[~outside] / tol
        stats = m.surrogate.profiling_stats
        return {"tol": tol, "evals": m.evals, "leaves": m.leaves, **error_stats(e), "worst": gm.r3(float(e.max())),
                "p99": gm.r3(float(np.quantile(e, 0.99))), "flagged": gm.r3(100 * float(outside.mean())),
                "train_s": gm.r3(probe.seconds), "oracle_s": gm.r3(oracle["s"]), "oracle_calls": oracle["calls"],
                "peak_host_mb": probe.peak_mb, "peak_device_mb": probe.device_mb,
                "predict_us": gm.r3(us), "flagged_us": gm.r3(flagged_us),
                "model_bytes": saved_bytes(lambda m, path: m.surrogate.save_model(path), m), "n_test": len(X), "n0": n0,
                "levels": int(stats["num_levels"]), "unresolved": int(stats["num_unresolved_clusters"])}  # fmt: skip

    rows = []
    for tol in tols:
        n0 = size(rows)
        if n0 is not None and n0 > MAX_SAMPLE:
            break
        key = {"kind": "baseline", "method": name, "data": data_hash(lo, hi, X, truth), "tol": repr(tol),
               "settings": {"n0": repr(n0), **{k: repr(v) for k, v in sorted(kw.items())}}}  # fmt: skip
        rows.append(STORE.get(key, case, tol, n0, changed=RESULTS_CHANGED, method=name))
        print(name, len(lo), rows[-1], flush=True)
    return rows


def grown_size(d: int, fixed: list[dict], grown: list[dict]) -> int:
    """
    The adaptive companion's grown sample (``grown_size`` in its convergence.py, whose constants this reads):
    POINTS_PER_LEAF points per expected leaf, the leaves extrapolated from the last count by its recent growth per
    halving, between 1 and 2^(d/2).
    """
    rows = grown if len(grown) >= 2 else fixed[: len(grown) + 1]
    counts = [r["leaves"] for r in rows[-4:]]
    growth = 1.0 if len(counts) < 2 else (counts[-1] / counts[0]) ** (1 / (len(counts) - 1))
    growth = min(max(growth, 1.0), 2 ** (d / 2))
    expected = fixed[0]["leaves"] if not grown else grown[-1]["leaves"] * growth
    return max(round(POINTS_PER_LEAF * expected), 200 * (d + 1))


def study_ladders(f, lo, hi, tols, X, truth, art_tols=None) -> dict:
    """AFINNI from the grown sample and ART (at ``art_tols``, or ``tols``) on the box, with the adaptive companion's convergence settings."""
    d = len(lo)
    afinni = {"num_initial_clusters": 4, "min_pts_per_cluster": 8 * (d + 1), "max_levels": 50}
    fixed = baseline_ladder("afinni", f, lo, hi, tols, X, truth, lambda rows: 200 * (d + 1), **afinni)
    return {"afinni": baseline_ladder("afinni", f, lo, hi, tols[: len(fixed)], X, truth,
                                      lambda rows: grown_size(d, fixed, rows), **afinni),
            "art": baseline_ladder("art", f, lo, hi, tols if art_tols is None else art_tols, X, truth, lambda rows: 20 * (d + 1), num_initial_clusters=1,
                                   max_levels=50, art=ARTSettings(sample_multiplier=20))}  # fmt: skip


def ridge_prediction(m, row) -> None:
    """ChebTT-M's degree per decade along the ridge, ln 10 / ln rho, from tanh's nearest pole in its rotated box (thm:decay)."""
    cols, Q, models = m.groups[0]
    d = Q.shape[0]
    q = gm.lift_direction(d)
    hw = -models[0].lo[0]
    if Q is None or abs(abs(Q[:, 0] @ q) - 1) > 1e-6 or abs(hw - np.abs(Q[:, 0]) @ np.ones(d)) > 1e-12:
        raise SystemExit("ChebTT-M's first rotated axis is not the ridge's direction, or its box is not |Q|^T half.")
    # q.x + offset = i pi / (2 S) is tanh's nearest pole; on the first axis, q.x = u_1 (q . Q_1)
    t = (-gm.RIDGE_OFFSET + 1j * np.pi / (2 * gm.RIDGE_STEEP)) / (q @ Q[:, 0]) / hw
    rho = max(abs(t + np.sqrt(t * t - 1)), abs(t - np.sqrt(t * t - 1)))
    row["rho"] = gm.r3(float(rho))
    row["per_decade"] = gm.r3(float(np.log(10) / np.log(rho)))


def leaf_rate(rows: list[dict], fit_leaves: int) -> float | None:
    """The adaptive study's fitted rate: the slope of log(leaves) against log(1 / tol) over the rungs with at least fit_leaves leaves."""
    use = [r for r in rows if r["leaves"] >= fit_leaves]
    if len(use) < 3:
        return None
    return gm.r3(float(np.polyfit(np.log([1 / r["tol"] for r in use]), np.log([r["leaves"] for r in use]), 1)[0]))


def per_decade(rows: list[dict]) -> float:
    """The least-squares slope of the ridge axis's degree against log10(1 / tol)."""
    ok = [r for r in rows if "degrees" in r]
    return gm.r3(float(np.polyfit([-np.log10(r["tol"]) for r in ok], [r["degrees"][0] for r in ok], 1)[0]))


def robertson():
    """
    Robertson's step map as the step-map suite poses it: the increment over one step, in log10 abundances (dex),
    on the padded box of the training trajectories' states; tested at the test trajectories' states.
    """
    from src.benchmark.step_maps import data, state_box, visited
    from src.problems.ode import STEP_MAPS

    sm = STEP_MAPS["robertson"]
    train, test = data("robertson")
    lo, hi = state_box(visited(train))

    def f(Z):
        return sm.step(Z) - Z

    X = visited(test)
    return f, lo, hi, X, f(X)


# ----------------------------------------------------------------------------- panels
MARK = {"chebtt": "s", "chebtt_metric": "D", "chebtt_adaptive": "^", "hub": "v", "afinni": "o", "art": "o"}
LABEL = {"chebtt": "ChebTT", "chebtt_metric": "ChebTT-M", "chebtt_adaptive": "ChebTT-A", "hub": "hub", "afinni": "AFINNI", "art": "ART"}


def evals_line(ax, rows, method, holds=lambda r: True) -> None:
    """Evaluations against 1 / tol; hollow where the model misses the tolerance, a cross where the build failed."""
    ok = [r for r in rows if "failed" not in r]
    if ok:
        x, y = [1 / r["tol"] for r in ok], [r["evals"] for r in ok]
        ax.loglog(x, y, "-", color=gm.METHOD[method], lw=0.9, zorder=2)
        for r, xi, yi in zip(ok, x, y, strict=True):
            filled = holds(r)
            ax.loglog(xi, yi, MARK[method], color=gm.METHOD[method], ms=2.8, mew=0.7,
                      mfc=gm.METHOD[method] if filled else "white", zorder=3)  # fmt: skip
    for r in rows:
        if "failed" in r:
            ax.loglog(1 / r["tol"], r["evals"], "x", color=gm.METHOD[method], ms=4, mew=1.0, zorder=4)


def axes_style(ax, title=None) -> None:
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel(r"$1 / \alpha$", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.minorticks_off()
    gm.tidy(ax)


def holds(r) -> bool:
    return r.get("worst", np.inf) <= 1.0


def problem_panel(name: str, data: dict, dims, methods, baselines: dict) -> None:
    fig, axes = plt.subplots(1, len(dims), figsize=(2.2 * len(dims), 1.9), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), dims, strict=True):
        for method in methods:
            evals_line(ax, data[str(d)][method], method, holds)
        for method, rows in baselines.get(d, {}).items():
            x, y = [1 / r["tol"] for r in rows], [r["evals"] for r in rows]
            ax.loglog(x, y, "o-", color=gm.METHOD[method], ms=2.2, lw=0.9, zorder=2)
        axes_style(ax, f"$d = {d}$")
    np.atleast_1d(axes)[0].set_ylabel("evaluations of $f$", fontsize=6.5, labelpad=1)
    fig.tight_layout(w_pad=0.5)
    gm.save(fig, f"convergence-{name}")


def vector_panel(vector: dict) -> None:
    """Evaluations against 1 / tol on the vector problems; hollow where a model misses, a cross where a build failed."""
    fig, axes = plt.subplots(1, len(vector), figsize=(2.2 * len(vector), 1.9), sharey=True)
    for ax, (name, row) in zip(np.atleast_1d(axes), vector.items(), strict=True):
        for method in SCHEMES:
            evals_line(ax, row[method], method, holds)
        for method in ("afinni", "art"):
            rows = row[method]
            ax.loglog([1 / r["tol"] for r in rows], [r["evals"] for r in rows], "o-", color=gm.METHOD[method], ms=2.2, lw=0.9, zorder=2)
        axes_style(ax, name)
    np.atleast_1d(axes)[0].set_ylabel("evaluations of $f$", fontsize=6.5, labelpad=1)
    fig.tight_layout(w_pad=0.5)
    gm.save(fig, "convergence-vector")


def degree_panel(ridge: dict) -> None:
    fig, ax = plt.subplots(figsize=(2.4, 1.9))
    for d, ls in zip(DIMS, ("-", "--", ":"), strict=True):
        rows = [r for r in ridge[str(d)]["chebtt_metric"] if "degrees" in r]
        x = np.array([-np.log10(r["tol"]) for r in rows])
        y = np.array([r["degrees"][0] for r in rows])
        ax.plot(x, y, "D", color=gm.METHOD["chebtt_metric"], ms=2.6, zorder=3)
        pred = rows[0]["per_decade"]
        b = np.mean(y - pred * x)  # the prediction is a slope: anchored by least squares
        ax.plot(x, b + pred * x, ls, color=gm.GREY, lw=0.9, zorder=2)
        ax.text(x[-1] + 0.25, y[-1], f"$d = {d}$", fontsize=5.5, va="center")
    ax.set_xlabel(r"$\log_{10}(1 / \alpha)$", fontsize=6.5, labelpad=1)
    ax.set_ylabel("degree along the ridge", fontsize=6.5, labelpad=1)
    ax.set_xlim(right=x[-1] + 1.6)
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    gm.tidy(ax)
    gm.save(fig, "convergence-degree")


def floor_panel(floor: dict) -> None:
    """The hub's worst absolute error against the tolerance it was asked for, beside the line error = tol."""
    fig, ax = plt.subplots(figsize=(2.4, 1.9))
    styles = {"rosen-3": ("-", gm.METHOD["hub"]), "rosen-5": ("--", gm.METHOD["hub"]),
              "shared_latent": ("-", gm.FEATURE["ridge"]), "lift": ("--", gm.FEATURE["ridge"])}  # fmt: skip
    for key, rows in floor.items():
        ls, colour = styles[key]
        x = [1 / r["tol"] for r in rows]
        ax.loglog(x, [r["worst"] * r["tol"] for r in rows], "v" + ls, color=colour, ms=2.6, lw=0.9, zorder=3)
    t = np.array([1e1, 1e6])
    ax.loglog(t, 1 / t, color=gm.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_ylabel(r"worst $|f - \hat f|$", fontsize=6.5, labelpad=1)
    axes_style(ax)
    gm.save(fig, "convergence-floor")


def feature_panel(F: dict) -> None:
    """On F: evaluations, and the worst error in units of tol, against 1 / tol."""
    fig, (ax_n, ax_e) = plt.subplots(1, 2, figsize=(4.4, 1.9))
    for method, rows in F.items():
        if method in ("afinni", "art"):
            x = [1 / r["tol"] for r in rows]
            ax_n.loglog(x, [r["evals"] for r in rows], "o-", color=gm.METHOD[method], ms=2.2, lw=0.9)
            ax_e.loglog(x, [r["worst"] for r in rows], "o-", color=gm.METHOD[method], ms=2.2, lw=0.9)
            continue
        evals_line(ax_n, rows, method, holds)
        ok = [r for r in rows if "failed" not in r]
        if ok:
            ax_e.loglog([1 / r["tol"] for r in ok], [r["worst"] for r in ok], MARK[method] + "-", color=gm.METHOD[method], ms=2.6, lw=0.9)
    ax_e.axhline(1.0, color=gm.GREY, lw=0.6, ls=(0, (3, 2)))
    ax_n.set_ylabel("evaluations of $f$", fontsize=6.5, labelpad=1)
    ax_e.set_ylabel(r"worst $|f - \hat f| / \alpha$", fontsize=6.5, labelpad=1)
    for ax in (ax_n, ax_e):
        axes_style(ax)
    fig.tight_layout(w_pad=1.0)
    gm.save(fig, "convergence-feature")


def every_case(out: dict):
    """(method, row) for every finished case of the study, each once (the hub's floor runs on Rosenbrock are its rosen rows)."""
    groups = [*out["rosen"].values(), *out["ridge"].values(), out["F"], *out["vector"].values()]
    groups.append({"hub": out["floor"]["shared_latent"] + out["floor"]["lift"]})
    for g in groups:
        for m in (*SCHEMES, "afinni", "art"):
            yield from ((m, r) for r in g.get(m, []) if isinstance(r, dict) and "failed" not in r)


def times_panel(out: dict) -> None:
    """Every case's training time against its evaluations, and its inference time per query; AFINNI's and ART's as circles."""
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(4.6, 1.9))
    for m, r in every_case(out):
        kw = {"color": gm.METHOD[m], "ms": 2.4, "mew": 0.6, "ls": "none", "zorder": 3, "alpha": 0.8}
        ax_t.loglog(r["evals"], r["train_s"], MARK[m], **kw)
        ax_p.loglog(r["evals"], r["predict_us"], MARK[m], **kw)
    ax_t.set_ylabel("training time (s)", fontsize=6.5, labelpad=1)
    ax_p.set_ylabel(r"inference time per query ($\mu$s)", fontsize=6.5, labelpad=1)
    for ax in (ax_t, ax_p):
        ax.set_xlabel("evaluations of $f$", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.minorticks_off()
        gm.tidy(ax)
    fig.tight_layout(w_pad=1.0)
    gm.save(fig, "convergence-times")


def time_summary(out: dict) -> dict:
    """Per method, over every case: the median training time per evaluation (us), and the inference time's range and median."""
    summary = {}
    for m in (*SCHEMES, "afinni", "art"):
        rows = [r for k, r in every_case(out) if k == m and "train_s" in r]
        per_eval = [1e6 * r["train_s"] / r["evals"] for r in rows]
        us = [r["predict_us"] for r in rows]
        summary[m] = {"cases": len(rows), "per_eval_us": gm.r3(float(np.median(per_eval))),
                      "predict_us": [min(us), gm.r3(float(np.median(us))), max(us)]}  # fmt: skip
    return summary


# ----------------------------------------------------------------------------- the study
def main() -> None:
    global POINTS_PER_LEAF, MAX_SAMPLE
    adaptive = json.loads(ADAPTIVE.read_text())
    POINTS_PER_LEAF, MAX_SAMPLE = adaptive["points_per_leaf"], adaptive["max_sample"]
    X2 = gm.grid(gm.R_LO, gm.R_HI, 40)
    if not np.array_equal(gm.rosenbrock(X2), (1.0 - X2[:, 0]) ** 2 + (X2[:, 1] - X2[:, 0] ** 2) ** 2):
        raise SystemExit("The chain at d = 2 is not the running example's Rosenbrock valley.")
    if adaptive["box"] != {"lower": [float(gm.R_LO[0]), float(gm.R_LO[1])], "upper": [float(gm.R_HI[0]), float(gm.R_HI[1])]}:
        raise SystemExit("The adaptive companion's convergence study ran on a different box.")
    lo, hi = rosen_box(2)  # compile every kernel the runs use, so no case's training time includes it
    for m in SCHEMES:
        per_query_us(gm.scheme_on(m, gm.rosenbrock, lo, hi, 0.1).predict, X2)
        per_query_us(gm.scheme_on(m, lambda x: np.c_[gm.rosenbrock(x), gm.rosenbrock(x)], lo, hi, 0.1, 2).predict, X2)
    for m in ("afinni", "art"):
        per_query_us(gm.baseline_on(m, lo, hi, 1.0, f=gm.rosenbrock).predict, X2)
    out: dict = {"decades": DECADES, "dims": list(DIMS), "rosen": {}, "ridge": {}}

    for d in DIMS:
        lo, hi = rosen_box(d)
        X = test_points(lo, hi)
        truth = gm.rosenbrock(X)
        out["rosen"][str(d)] = {m: scheme_ladder(m, gm.rosenbrock, lo, hi, DECADES["rosen"], X, truth) for m in SCHEMES}

    for d in DIMS:
        lo, hi = -np.ones(d), np.ones(d)
        X = test_points(lo, hi)
        truth = gm.lift_ridge(X)
        row = {m: scheme_ladder(m, gm.lift_ridge, lo, hi, DECADES["ridge"], X, truth,
                                extra=ridge_prediction if m == "chebtt_metric" else None) for m in SCHEMES}  # fmt: skip
        tol0, n_afinni, n_art = RIDGE_HALVINGS[d]
        row.update(study_ladders(gm.lift_ridge, lo, hi, [tol0 / 2**k for k in range(n_afinni)], X, truth,
                                 [tol0 / 2**k for k in range(n_art)]))  # fmt: skip
        out["ridge"][str(d)] = row
    out["ridge_degree"] = {str(d): {"predicted": out["ridge"][str(d)]["chebtt_metric"][0]["per_decade"],
                                    "measured": per_decade(out["ridge"][str(d)]["chebtt_metric"]),
                                    "rho": out["ridge"][str(d)]["chebtt_metric"][0]["rho"]} for d in DIMS}  # fmt: skip

    out["ridge_rates"] = {str(d): {m: leaf_rate(out["ridge"][str(d)][m], adaptive["fit_leaves"]) for m in ("afinni", "art")}
                          for d in DIMS}  # fmt: skip

    X = gm.grid(gm.F_LO, gm.F_HI, 300)
    truth = gm.LAYERS["brj"](X)
    F = {m: scheme_ladder(m, gm.LAYERS["brj"], gm.F_LO, gm.F_HI, DECADES["F"], X, truth) for m in SCHEMES}
    tol0, n = F_HALVINGS
    for m in ("afinni", "art"):  # this companion's AFINNI and ART on F (section 7): baseline_on's settings
        F[m] = baseline_ladder(m, gm.LAYERS["brj"], gm.F_LO, gm.F_HI, [tol0 / 2**k for k in range(n)], X, truth, lambda rows: None)
    out["F"] = F

    problem = FUNCTIONS[gm.HUB_PROBLEM]
    lo, hi = problem.domain(HUB_D)
    T = problem.test_points(HUB_D)
    floor = {"rosen-3": out["rosen"]["3"]["hub"], "rosen-5": out["rosen"]["5"]["hub"],
             "shared_latent": scheme_ladder("hub", problem.values, lo, hi, DECADES["hub"], T, problem.values(T))}  # fmt: skip
    Xl = np.random.default_rng(20_261_117).uniform(-1, 1, (20_000, HUB_D))
    floor["lift"] = scheme_ladder("hub", gm.lift_hub, -np.ones(HUB_D), np.ones(HUB_D), DECADES["hub"], Xl, gm.lift_hub(Xl))
    out["floor"] = floor

    # A vector function and a flow map, AFINNI's and ART's runs on them the adaptive study's; and the stiff step
    # map, whose AFINNI and ART run here.
    out["vector"] = {}
    for name, (d, tols) in VECTOR.items():
        p = FUNCTIONS[name]
        lo, hi = p.domain(d)
        X = p.test_points(d)
        truth = p.values(X)
        row = {m: scheme_ladder(m, p.values, lo, hi, tols, X, truth, outputs=truth.shape[1]) for m in SCHEMES}
        study = adaptive["problems"][name]["runs"]
        row.update({"afinni": study["afinni_scaled"], "art": study["art"]})
        out["vector"][name] = {"d": d, "outputs": int(truth.shape[1]), **row}
    f, lo, hi, X, truth = robertson()
    row = {m: scheme_ladder(m, f, lo, hi, STIFF[0], X, truth, outputs=truth.shape[1]) for m in SCHEMES}
    row.update(study_ladders(f, lo, hi, STIFF[1], X, truth, STIFF[2]))
    out["vector"]["robertson"] = {"d": len(lo), "outputs": int(truth.shape[1]), **row}
    vector_panel(out["vector"])

    baselines = {d: {m: adaptive["runs"]["afinni_scaled" if m == "afinni" else m][str(d)] for m in ("afinni", "art")}
                 for d in DIMS if str(d) in adaptive["runs"]["art"]}  # fmt: skip
    problem_panel("rosen", out["rosen"], DIMS, SCHEMES, baselines)
    problem_panel("ridge", out["ridge"], DIMS, SCHEMES, {d: {m: out["ridge"][str(d)][m] for m in ("afinni", "art")} for d in DIMS})
    degree_panel(out["ridge"])
    floor_panel(floor)
    feature_panel(F)
    times_panel(out)
    out["times"] = time_summary(out)
    out["provenance"] = provenance()
    (OUT / "convergence.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
