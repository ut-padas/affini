"""Draw the panels of the companion's appendix on the batched ODE integrator (apx-ode-*.svg, apx-ode.json).

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/report/figures/appendix/ode.py

One batch of the manuscript study's Lorenz-96 oracle at d = 5 (the first rows of its fixed test set,
``studies/manuscript/case.py``) is integrated three ways: by the library's ``integrate_ensemble`` (the
Numba CPU twin of ``ode.cu``), by SciPy's ``solve_ivp(method="RK45")`` once per row, and by one
``solve_ivp`` over the stacked batch. The Numba twin does not expose its steps, so the script restates
``_trajectory_cpu`` in Python, recording every attempted step, and stops unless the restatement ends
on the library's bits for every row, the library equals the study's oracle, and every row takes
SciPy's accepted steps.
"""

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src import IS_CUPY_BACKEND  # noqa: E402
from src.numerics import DOPRI5, integrate_ensemble  # noqa: E402
from src.numerics.ode import ODE_BLOCK, ODE_THREAD_MAX_D  # noqa: E402
from src.problems.lorenz96 import lorenz96  # noqa: E402
from src.problems.lorenz96_flowmap import LORENZ96_ATOL, LORENZ96_RTOL, lorenz96_flowmap  # noqa: E402
from studies.manuscript.case import functions, test_points  # noqa: E402
from studies.manuscript.spec import FUNCTIONS  # noqa: E402

D = 5
BATCH = 64
SHOWN = 4  # trajectories drawn in the step-size panel
SUBSETS = (1, 7, BATCH)  # batch sizes integrate_ensemble must give the same rows in
STACKS = (1, 8, BATCH)  # stacked solve_ivp batches the first row is integrated in
_defaults = inspect.signature(lorenz96_flowmap).parameters
FORCING, TF = float(_defaults["forcing"].default), float(_defaults["tf"].default)


def rhs_component(x: list[float], j: int, d: int) -> float:
    """``_lorenz96_rhs`` of ode.py, in the same order of operations."""
    jp1 = 0 if j + 1 == d else j + 1
    jm1 = d - 1 if j == 0 else j - 1
    jm2 = j - 2 if j >= 2 else j - 2 + d
    return (x[jp1] - x[jm2]) * x[jm1] - x[j] + FORCING


def restated_trajectory(y0: np.ndarray, t1: float, rtol: float, atol: float) -> tuple[list[float], list[tuple]]:
    """``_trajectory_cpu`` of ode.py in Python floats (IEEE double, no fused operations), recording ``(t, h, accepted)`` per attempt."""
    A, b, e, order, s = DOPRI5.A.tolist(), DOPRI5.b.tolist(), DOPRI5.e.tolist(), DOPRI5.order, DOPRI5.A.shape[0]
    d = len(y0)
    y = [float(v) for v in y0]
    k = [[0.0] * d for _ in range(s)]
    for j in range(d):
        k[0][j] = rhs_component(y, j, d)
    sy = sf = 0.0
    for j in range(d):
        sc = atol + rtol * abs(y[j])
        r, q = y[j] / sc, k[0][j] / sc
        sy += r * r
        sf += q * q
    d0, d1 = math.sqrt(sy / d), math.sqrt(sf / d)
    h0 = min(1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1, t1)
    x = [y[j] + h0 * k[0][j] for j in range(d)]
    sd = 0.0
    for j in range(d):
        r = (rhs_component(x, j, d) - k[0][j]) / (atol + rtol * abs(y[j]))
        sd += r * r
    d2 = math.sqrt(sd / d) / h0
    dm = max(d1, d2)
    h1 = max(1e-6, h0 * 1e-3) if dm <= 1e-15 else (0.01 / dm) ** (1.0 / order)
    h = min(min(100.0 * h0, h1), t1)
    t, rejected, log = 0.0, False, []
    while t < t1:
        min_step = 10.0 * (math.nextafter(t, math.inf) - t)
        if h < min_step:
            if rejected:
                raise SystemExit("A study trajectory failed in the restatement.")
            h = min_step
        last = t + h >= t1
        if last:
            h = t1 - t
        for i in range(1, s):
            for j in range(d):
                acc = 0.0
                for q in range(i):
                    acc += A[i][q] * k[q][j]
                x[j] = y[j] + h * acc
            for j in range(d):
                k[i][j] = rhs_component(x, j, d)
        se, yn = 0.0, [0.0] * d
        for j in range(d):
            accb = acce = 0.0
            for q in range(s):
                accb += b[q] * k[q][j]
                acce += e[q] * k[q][j]
            yn[j] = y[j] + h * accb
            r = (h * acce) / (atol + rtol * max(abs(y[j]), abs(yn[j])))
            se += r * r
        err = math.sqrt(se / d)
        log.append((t, h, err < 1.0))
        if err < 1.0:
            t = t1 if last else t + h
            y = yn
            k[0] = list(k[s - 1])
            factor = 10.0 if err == 0.0 else min(10.0, 0.9 * err ** (-1.0 / order))
            h *= min(1.0, factor) if rejected else factor
            rejected = False
        else:
            h *= max(0.2, 0.9 * err ** (-1.0 / order))
            rejected = True
    return y, log


def scipy_alone(x0: np.ndarray):
    return solve_ivp(lambda t, x: lorenz96(x[None], FORCING)[0], (0.0, TF), x0, method="RK45",
                     rtol=LORENZ96_RTOL, atol=LORENZ96_ATOL)  # fmt: skip


def scipy_stacked(X0: np.ndarray):
    n = X0.shape[0]
    return solve_ivp(lambda t, x: lorenz96(x.reshape(n, -1), FORCING).ravel(), (0.0, TF), X0.ravel(), method="RK45",
                     rtol=LORENZ96_RTOL, atol=LORENZ96_ATOL)  # fmt: skip


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    if IS_CUPY_BACKEND:
        raise SystemExit("Run on the CPU: the script restates the Numba twin of the kernel.")
    domain = FUNCTIONS["lorenz96"].domain
    X = test_points("lorenz96", D, domain, BATCH)
    Y = integrate_ensemble("lorenz96", X, TF, [FORCING], LORENZ96_RTOL, LORENZ96_ATOL)
    oracle, _ = functions("lorenz96")
    check(np.array_equal(oracle(X), Y[:, :1]), "integrate_ensemble with these settings is not the study's oracle.")

    # 1. The restatement ends on the library's bits; each row's accepted steps are SciPy's.
    logs, h_rel, t_gap, sci_gap, steps = [], 0.0, 0.0, 0.0, []
    for i in range(BATCH):
        y, log = restated_trajectory(X[i], TF, LORENZ96_RTOL, LORENZ96_ATOL)
        check(np.array_equal(np.array(y), Y[i]), f"The restated integrator differs from the library on row {i}.")
        sol = scipy_alone(X[i])
        ours = np.array([h for _, h, ok in log if ok])
        theirs = np.diff(sol.t)
        check(len(ours) == len(theirs), f"Row {i} takes {len(ours)} steps, SciPy {len(theirs)}.")
        # The last step is t1 - t, which cancels, so it is compared through the step times instead.
        h_rel = max(h_rel, float(np.max(np.abs(ours[:-1] - theirs[:-1]) / theirs[:-1], initial=0.0)))
        t_gap = max(t_gap, float(np.max(np.abs(np.cumsum(ours) - np.cumsum(theirs)))) / TF)
        sci_gap = max(sci_gap, float(np.max(np.abs(Y[i] - sol.y[:, -1]))))
        logs.append(log)
        steps.append(len(ours))
    # The error estimate sum_q e_q k_q cancels (the e_q sum to zero), so its rounding, and the step
    # sizes taken from it, agree to about 1e-9 relative rather than to the unit roundoff.
    check(h_rel < 1e-7 and t_gap < 1e-7, "The library's steps are not SciPy's up to rounding.")
    steps = np.array(steps)
    rejections = np.array([sum(not ok for _, _, ok in log) for log in logs])

    # 2. A row does not depend on its batch (bit for bit); a stacked solve_ivp does.
    perm = np.random.default_rng(mf.SEED).permutation(BATCH)
    for size in SUBSETS:
        rows = perm[:size]
        check(np.array_equal(integrate_ensemble("lorenz96", X[rows], TF, [FORCING], LORENZ96_RTOL, LORENZ96_ATOL), Y[rows]),
              "integrate_ensemble changed a row with its batch.")  # fmt: skip
    alone = np.array([scipy_alone(X[i]).y[:, -1] for i in range(BATCH)])
    stacked = scipy_stacked(X)
    stacked_rows = stacked.y[:, -1].reshape(BATCH, D)
    stacked_gap = np.max(np.abs(stacked_rows - alone), axis=1)
    library_gap = np.max(np.abs(Y - alone), axis=1)
    row0 = {size: scipy_stacked(X[:size]).y[:, -1].reshape(size, D)[0] for size in STACKS}
    row0_spread = max(float(np.max(np.abs(row0[a] - row0[1]))) for a in row0)

    # 3. Panels: h(t) of a few rows beside the stacked solve's shared step; then each row's gap to SciPy.
    order = np.argsort(steps, kind="stable")
    shown = [int(order[0]), int(order[BATCH // 3]), int(order[2 * BATCH // 3]), int(order[-1])][:SHOWN]
    colours = [mf.PALETTE[c] for c in (3, 5, 2, 6)]
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    for i, c in zip(shown, colours, strict=True):
        acc = [(t, h) for t, h, ok in logs[i] if ok]
        ts = np.array([t for t, _ in acc] + [TF])
        hs = np.array([h for _, h in acc] + [acc[-1][1]])
        ax.step(ts, hs, where="post", color=c, lw=1.1)
        sol = scipy_alone(X[i])
        ax.plot(sol.t[:-1], np.diff(sol.t), "o", ms=2.6, mfc="white", mec=c, mew=0.6, ls="none")
        rej = [(t, h) for t, h, ok in logs[i] if not ok]
        if rej:
            ax.plot(*np.array(rej).T, "x", color=c, ms=3.5, mew=0.8)
    st = np.diff(stacked.t)
    ax.step(np.r_[stacked.t[:-1], TF], np.r_[st, st[-1]], where="post", color=mf.DARK, lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(0, TF)
    ax.set_xlabel("$t$")
    ax.set_ylabel("step size $h$")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    mf.save(fig, "apx-ode-steps")

    floor = 1e-18  # an exact agreement is drawn here, on the tick labelled 0
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    xs = np.arange(BATCH)
    ax.semilogy(xs, stacked_gap[order], "o", ms=2.6, color=mf.GREY, label="one solve_ivp over the batch")
    ax.semilogy(xs, np.maximum(library_gap[order], floor), "o", ms=2.6, color=mf.DARK, label="integrate_ensemble")
    for i, c in zip(shown, colours, strict=True):
        p = int(np.nonzero(order == i)[0][0])
        ax.semilogy([p], [stacked_gap[i]], "o", ms=4.2, mfc="none", mec=c, mew=1.0)
        ax.semilogy([p], [max(library_gap[i], floor)], "o", ms=4.2, mfc="none", mec=c, mew=1.0)
    ax.axhline(LORENZ96_ATOL, color=mf.LIGHT, lw=0.8, ls="--", label="the solver's tolerance")
    ax.set_xlabel("rows of the batch, by their own number of steps")
    ax.set_ylabel(r"$\max_j |x_j(t_1) - x_j^{\mathrm{RK45\ alone}}(t_1)|$")
    ax.set_ylim(floor / 3, LORENZ96_ATOL * 30)
    ticks = [floor] + [10.0**e for e in range(-17, -4, 3)]
    ax.set_yticks(ticks, ["0"] + [f"$10^{{{int(round(np.log10(t)))}}}$" for t in ticks[1:]])
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=6, loc="center left")
    mf.save(fig, "apx-ode-error")

    out = {
        "d": D, "batch": BATCH, "tf": TF, "tol_text": f"{LORENZ96_RTOL:.0e}", "forcing": FORCING, "rtol": LORENZ96_RTOL, "atol": LORENZ96_ATOL,
        "domain": list(domain), "steps_min": int(steps.min()), "steps_max": int(steps.max()),
        "rejections_max": int(rejections.max()), "rows_rejected": int(np.sum(rejections > 0)),
        "stacked_steps": int(len(stacked.t) - 1),
        "shown_steps": [int(steps[i]) for i in shown], "shown_colours": colours,
        "h_rel_max": f"{h_rel:.0e}", "t_gap_max": f"{t_gap:.0e}", "scipy_gap_max": f"{sci_gap:.0e}", "library_gap_max": f"{float(library_gap.max()):.0e}",
        "stacked_gap_max": f"{float(stacked_gap.max()):.0e}", "stacked_gap_min": f"{float(stacked_gap.min()):.0e}",
        "row0_spread": f"{row0_spread:.0e}",
        "thread_max_d": ODE_THREAD_MAX_D, "warp_max_d": 32 * ODE_THREAD_MAX_D, "block": ODE_BLOCK,
        "order": DOPRI5.order, "warp": 32, "subsets": list(SUBSETS), "stacks": list(STACKS), "stages": int(DOPRI5.A.shape[0]),
    }
    (mf.OUT / "apx-ode.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
