"""Draw the panels of the appendix section on the soft search's monitored descent and write apx-soft.json.

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/report/figures/appendix/soft.py

The running example's ART run is trained through make_figures (``record_art``), with
``src.art.plan.soft_objective`` wrapped to record the library's objective at every iteration. Each
split node's soft search is then replayed with the library's own primitives (``soft_objective``,
``soft_gates``, ``unit_rows``, ``plane_scores``, ``hard_sse``) in the order of
``ARTPlan._soft_refine``, recording every trial of every line search. The script stops unless the
replay ends at the plane the library stored for every node, its objective values equal the
library's, its stop reasons add up to the plan's counters, and the gap identity of the companion
holds on the recorded node.
"""

import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf  # noqa: E402
import numpy as np  # noqa: E402
from make_figures import DARK, GREY, RED, plt  # noqa: E402

import src.art.plan as art_plan_module  # noqa: E402
from src.art.geometry import plane_scores  # noqa: E402
from src.art.plan import ARTPlan  # noqa: E402
from src.art.split_search import (  # noqa: E402
    SplitRows,
    hard_sse,
    soft_gates,
    soft_objective,
    unit_rows,
)
from src.kernels import source as kernel_source  # noqa: E402
from src.numerics import det_einsum  # noqa: E402

EPS = float(np.finfo(np.float64).eps)
REASONS = ("soft_nonfinite", "soft_gate_saturated", "soft_gradient_converged", "soft_line_search_stalled",
           "soft_iteration_limit")  # fmt: skip


def constants() -> dict:
    """The line search's constants, read from ``_soft_refine`` and checked against the GPU kernels."""
    src = inspect.getsource(ARTPlan._soft_refine)
    cu = kernel_source("art_split_search")
    raw = {
        "halvings": re.search(r"for _ in range\((\d+)\):", src).group(1),
        "armijo": re.search(r"Ft <= J - ([0-9.e-]+) \* alpha \* gn2", src).group(1),
        "rounding": re.search(r"margin = gap \+ (\d+) \* eps", src).group(1),
        "saturation": re.search(r"sp_sum / n < ([0-9.e-]+)\)", src).group(1),
    }
    if (f"soft_step(rows, w, z, g_w, g_z, J, gap, pen, rL, rR, sp_sum, T, n, delta, active, stops, {raw['halvings']})" not in src
            or f"J[k] - {raw['armijo']} * a * gn2[k]" not in cu
            or f"margin[k] = gap[k] + {raw['rounding']} * eps" not in cu
            or f"sp_sum[k] / n[k] < {raw['saturation']}" not in cu
            or "alpha = alpha * 0.5" not in src):  # fmt: skip
        raise SystemExit(
            "The soft search's constants differ between the CPU loop and the GPU kernels."
        )
    found = {"halvings": int(raw["halvings"]), "rounding": int(raw["rounding"]), "armijo": float(raw["armijo"]),
             "saturation": float(raw["saturation"])}  # fmt: skip
    return found


def rows_of(rec, ridge: float) -> SplitRows:
    return SplitRows(
        rec.U,
        rec.Y,
        rec.Phi,
        np.zeros(len(rec.U), dtype=np.int64),
        np.array([len(rec.U)]),
        ridge=ridge,
    )


def replay(rec, plan: ARTPlan, ridge: float, c: dict) -> dict:
    """``_soft_refine`` on one node, restated step by step on the library's primitives, with every trial kept."""
    s_cfg, rows = plan.settings, rows_of(rec, ridge)
    delta, n = s_cfg.soft_weight_floor, np.array([float(len(rec.U))])
    w, z = rec.bins[0][None].copy(), np.array([rec.bins[1]])
    s0 = plane_scores(rows.U, w[rows.labels])
    mean = rows.reduce(s0) / n
    var = rows.reduce((s0 - mean[rows.labels]) ** 2) / n
    T = s_cfg.soft_temperature * np.sqrt(
        np.maximum(var, (64 * EPS * np.maximum(np.abs(mean), 1e-300)) ** 2)
    )
    its, reason = [], "soft_iteration_limit"
    for _ in range(s_cfg.soft_iters):
        J, g_w, g_z, rL, rR, pen, gap, sp_sum = soft_objective(rows, w, z, T, delta)
        g_w = g_w - np.sum(g_w * w, axis=1, keepdims=True) * w
        gn2 = np.sum(g_w**2, axis=1) + g_z**2
        alpha = T / np.sqrt(np.maximum(gn2, 1e-300))
        margin = gap + c["rounding"] * EPS * (np.abs(J) + 1e-300)
        finite = bool(np.isfinite(J[0]) & np.isfinite(gap[0]) & np.isfinite(gn2[0]))
        trials, taken = [], False
        for m in range(c["halvings"]):
            w_t, z_t = unit_rows(w - alpha[:, None] * g_w), z - alpha * g_z
            WL, WR, _ = soft_gates(rows, w_t, z_t, T, delta)
            Ft = (rows.reduce(WL * rL + WR * rR) + pen) / n
            ok = (
                finite
                and bool(Ft[0] <= J[0] - c["armijo"] * alpha[0] * gn2[0])
                and bool(J[0] - Ft[0] > margin[0])
            )
            trials.append({"m": m, "dF": float(Ft[0] - J[0]), "armijo": float(c["armijo"] * alpha[0] * gn2[0]),
                           "accepted": ok})  # fmt: skip
            if ok:
                w, z, taken = w_t, z_t, True
                break
            alpha = alpha * 0.5
        its.append({"J": float(J[0]), "gap": float(gap[0]), "margin": float(margin[0]), "gn2": float(gn2[0]),
                    "step": float(T[0] * np.sqrt(gn2[0])), "trials": trials})  # fmt: skip
        if not taken:
            saturated = finite and sp_sum[0] / n[0] < c["saturation"]
            flat = finite and not saturated and T[0] * np.sqrt(gn2[0]) <= margin[0]
            reason = REASONS[0 if not finite else 1 if saturated else 2 if flat else 3]
            break
    # Harden as the library does: z moves to the first right-side score, kept only if the hard SSE improves.
    s = plane_scores(rows.U, w[rows.labels])
    z_h = s[s >= z[0]].min()
    sse = float(hard_sse(rows, w, np.array([z_h]), plan.min_side)[0][0])
    better = sse < rec.bins[2]
    final = (w[0], float(z_h)) if better else (rec.bins[0], rec.bins[1])
    if not (np.array_equal(final[0], rec.final[0]) and final[1] == rec.final[1]):
        raise SystemExit(
            "The replayed soft search does not end at the plane the library chose."
        )
    return {
        "its": its,
        "reason": reason,
        "T": float(T[0]),
        "sse_final": sse,
        "better": bool(better),
    }


def check_gap_identity(rec, plan: ARTPlan, ridge: float, node: dict) -> dict:
    """
    F(theta, B) - min_B F = (1/n) sum_s tr(R_s^T (H_s + lam I)^-1 R_s) for any B, checked at a
    perturbed B (where both sides are large); and the residual and sum forms of R at the computed B.
    """
    rows, delta = rows_of(rec, ridge), plan.settings.soft_weight_floor
    n = float(len(rec.U))
    w, z = rec.bins[0][None], np.array([rec.bins[1]])
    T, it0 = np.array([node["T"]]), node["its"][0]
    WL, WR, _ = soft_gates(rows, w, z, T, delta)
    lam = rows.lam[:, None, None]

    def terms(Bs: list) -> tuple[float, float]:
        """F(theta, B) and the trace form at the models ``Bs`` (one per child)."""
        F = gap = 0.0
        for Wt, B in zip((WL, WR), Bs, strict=True):
            H, _ = rows.gram(Wt)
            E = rows.Y - det_einsum(
                "np,npf->nf", rows.Phi, B[rows.labels]
            )  # soft_objective's form
            F += float(np.sum(Wt * np.sum(E**2, axis=1)) + rows.lam[0] * np.sum(B**2))
            R = rows.cross(rows.Phi, E, Wt) - lam * B
            gap += float(np.sum(R * rows.solve(H, R)))
        return F / n, gap / n

    B_t = [
        rows.solve(*rows.gram(Wt)) for Wt in (WL, WR)
    ]  # the computed models, as soft_objective forms them
    F_t, gap_t = terms(B_t)
    if not (
        np.isclose(F_t, it0["J"], rtol=1e-13, atol=0)
        and np.isclose(gap_t, it0["gap"], rtol=1e-12, atol=0)
    ):
        raise SystemExit(
            f"The restated soft objective or gap differs from the library's: {F_t} {it0['J']} {gap_t} {it0['gap']}"
        )
    rng = np.random.default_rng(mf.SEED)
    B_p = [B + 0.05 * rng.standard_normal(B.shape) for B in B_t]
    F_p, gap_p = terms(B_p)
    # min_B F cancels: F(B_p) - F(B_t) = gap(B_p) - gap(B_t), with both differences far above rounding.
    if not np.isclose(F_p - F_t, gap_p - gap_t, rtol=1e-9, atol=0):
        raise SystemExit(
            "The gap identity F(theta, B) - J_h(theta) = (1/n) tr(R^T M^-1 R) does not hold."
        )
    return {"perturbed_excess": F_p - F_t}


def gap_forms(
    rec, plan: ARTPlan, ridge: float, node: dict
) -> list[tuple[float, float, float]]:
    """
    Per child at the bins plane: the gap with R from the data residual (the library's form), with
    R from the sums, Phi^T W Y - (H + lam I) B, and in 60-digit arithmetic from the same float64
    inputs and computed models, which is the value both forms estimate.
    """
    import mpmath as mp

    mp.mp.dps = 60
    rows, delta = rows_of(rec, ridge), plan.settings.soft_weight_floor
    n, (N, p), fd = float(len(rec.U)), rows.Phi.shape, rows.Y.shape[1]
    WL, WR, _ = soft_gates(
        rows, rec.bins[0][None], np.array([rec.bins[1]]), np.array([node["T"]]), delta
    )
    lam = rows.lam[:, None, None]
    out = []
    for Wt in (WL, WR):
        H, Bs = rows.gram(Wt)
        B = rows.solve(H, Bs)
        E = rows.Y - det_einsum("np,npf->nf", rows.Phi, B[rows.labels])
        R_res = rows.cross(rows.Phi, E, Wt) - lam * B
        R_sum = Bs - (H + lam * np.eye(p)) @ B
        forms = [float(np.sum(R * rows.solve(H, R))) / n for R in (R_res, R_sum)]
        P, Y, Bm = (mp.matrix(a.tolist()) for a in (rows.Phi, rows.Y, B[0]))
        M, b = mp.zeros(p, p), mp.zeros(p, fd)
        for i in range(N):
            wi = mp.mpf(float(Wt[i]))
            for j in range(p):
                for k in range(p):
                    M[j, k] += P[i, j] * wi * P[i, k]
                for f in range(fd):
                    b[j, f] += P[i, j] * wi * Y[i, f]
        M += mp.mpf(float(rows.lam[0])) * mp.eye(p)
        R = b - M * Bm
        X = mp.lu_solve(M, R)
        exact = float(sum(R[j, f] * X[j, f] for j in range(p) for f in range(fd)) / n)
        out.append((*forms, exact))
    return out


def run_panel(node: dict, sse_bins: float, n: int) -> None:
    """J_t of every iteration above the number of trials its line search evaluated."""
    its = node["its"]
    t = np.arange(len(its))
    J = np.array([i["J"] for i in its])
    tried = np.array([len(i["trials"]) for i in its])
    took = np.array([i["trials"][-1]["accepted"] for i in its])
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(3.2, 2.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]}
    )
    ax.plot(
        t,
        J,
        color=DARK,
        lw=0.9,
        marker="o",
        ms=2.2,
        zorder=3,
        label="soft objective $J_h(\\theta_t)$",
    )
    ax.axhline(sse_bins / n, color=GREY, lw=0.7, ls="--", zorder=1)
    ax.axhline(node["sse_final"] / n, color=GREY, lw=0.7, ls=":", zorder=1)
    ax.text(
        t[-1],
        sse_bins / n,
        "hard error, bins plane",
        fontsize=6.5,
        color=GREY,
        ha="right",
        va="bottom",
    )
    ax.text(
        t[-1],
        node["sse_final"] / n,
        "hard error, final plane",
        fontsize=6.5,
        color=GREY,
        ha="right",
        va="bottom",
    )
    ax.set_ylim(node["sse_final"] / n - 0.03, J.max() + 0.03)
    ax.set_ylabel("mean squared error / $\\alpha_{tol}^2$", fontsize=7, labelpad=1)
    ax.legend(
        fontsize=7,
        frameon=False,
        loc="upper right",
        handlelength=1.4,
        borderaxespad=0.1,
    )
    bx.bar(t[took], tried[took], width=0.7, color=DARK, lw=0)
    bx.bar(t[~took], tried[~took], width=0.7, color=RED, lw=0)
    bx.set_ylabel("trials", fontsize=7, labelpad=1)
    bx.set_xlabel("iteration $t$", fontsize=7, labelpad=1)
    bx.set_yticks([1, len(its[-1]["trials"])])
    for a in (ax, bx):
        a.tick_params(labelsize=6.5, length=2, pad=1)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2, h_pad=0.3)
    mf.save(fig, "apx-soft-run")


def stop_panel(node: dict) -> None:
    """The trials of the last two iterations: each trial's change of the frozen objective, and the bar it must clear."""
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.5), sharey=True)
    lin = 1e-14
    for ax, k in zip(axes, (len(node["its"]) - 2, len(node["its"]) - 1), strict=True):
        it = node["its"][k]
        m = np.array([tr["m"] for tr in it["trials"]])
        dF = np.array([tr["dF"] for tr in it["trials"]])
        acc = np.array([tr["accepted"] for tr in it["trials"]])
        bar = -np.maximum(it["margin"], np.array([tr["armijo"] for tr in it["trials"]]))
        ax.fill_between(m, bar, -1, color="#e9edf3", lw=0, zorder=0)
        ax.plot(m, bar, color=GREY, lw=0.8, ls="--", zorder=1)
        ax.plot(
            [m[0], m[-1]], [-it["margin"]] * 2, color=GREY, lw=0.8, ls=":", zorder=1
        )
        ax.axhline(0, color=GREY, lw=0.5, zorder=1)
        ax.scatter(
            m[~acc],
            dF[~acc],
            s=12,
            facecolors="white",
            edgecolors=RED,
            lw=0.8,
            zorder=3,
        )
        ax.scatter(m[acc], dF[acc], s=16, c=DARK, lw=0, zorder=4)
        ax.set_yscale("symlog", linthresh=lin, linscale=0.6)
        ax.set_ylim(-1e-6, 1e-1)
        ax.set_xlim(-0.6, 11.6)
        ax.set_xticks([0, 4, 8, 11])
        ax.set_title(
            f"iteration {k}" + (": stalled" if not acc.any() else ""), fontsize=7, pad=2
        )
        ax.set_xlabel("halving $m$", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6.5, length=2, pad=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks([-1e-6, -1e-10, 0, 1e-10, 1e-6, 1e-2])
    axes[0].set_ylabel(
        "$F(\\theta', \\tilde B_t) - F(\\theta_t, \\tilde B_t)$", fontsize=7, labelpad=1
    )
    axes[0].text(0.3, -2e-7, "accepted below", fontsize=6.3, color=DARK, va="center")
    last = node["its"][-1]
    axes[1].text(
        0.2,
        -0.04 * last["trials"][0]["armijo"],
        "Armijo",
        fontsize=6.3,
        color=GREY,
        ha="left",
        va="center",
    )
    axes[1].text(
        11,
        -1.5 * last["margin"],
        "margin",
        fontsize=6.3,
        color=GREY,
        ha="right",
        va="top",
    )
    fig.tight_layout(pad=0.2, w_pad=0.6)
    mf.save(fig, "apx-soft-stop")


def main() -> None:
    if mf.IS_CUPY_BACKEND:
        raise SystemExit(
            "Run with SURROGATE_DEVICE=cpu: the panels and their numbers are the CPU results."
        )
    c = constants()
    library: list[
        tuple
    ] = []  # per soft_objective call: each node's first point, J and gap
    wrapped = art_plan_module.soft_objective

    def soft_objective_recorded(rows, w, z, T, delta):
        out = wrapped(rows, w, z, T, delta)
        library.append(
            (mf.to_host(rows.U[rows.starts]).copy(), mf.to_host(out[0]).copy())
        )
        return out

    art_plan_module.soft_objective = soft_objective_recorded
    try:
        _, _, searches, plan = mf.record_art()
    finally:
        art_plan_module.soft_objective = wrapped
    ridge = (
        1e-8 if plan.settings.leaf_fit == "tikhonov" else 0.0
    )  # as ARTPlan._split_rows
    nodes = {code: replay(rec, plan, ridge, c) for code, rec in searches.items()}
    counted = {r: sum(v["reason"] == r for v in nodes.values()) for r in REASONS}
    if any(counted[r] != plan.counters[r] for r in REASONS) or \
            sum(v["better"] for v in nodes.values()) != plan.counters["soft_improved"]:  # fmt: skip
        raise SystemExit(
            "The replayed stop reasons do not add up to the plan's counters."
        )

    ex = json.loads((mf.OUT / "example.json").read_text())
    code = int(ex["art"]["split"]["code"], 2)
    if code not in searches:
        raise SystemExit(
            "The companion's recorded split node has no recorded split search."
        )
    rec, node = searches[code], nodes[code]
    # The library's objective at this node, iteration by iteration, is the replay's to the bit.
    J_lib = [J[np.nonzero(np.all(rec.U[0] == U, axis=1))[0]] for U, J in library]
    J_lib = [float(j[0]) for j in J_lib if len(j) == 1]
    if J_lib[: len(node["its"])] != [i["J"] for i in node["its"]]:
        raise SystemExit(
            "The replayed soft objective differs from the library's at the recorded node."
        )
    check_gap_identity(rec, plan, ridge, node)
    forms = np.array(
        [
            g
            for k in sorted(nodes)
            for g in gap_forms(searches[k], plan, ridge, nodes[k])
        ]
    )
    res_err = np.abs(forms[:, 0] / forms[:, 2] - 1)
    sum_off = (forms[:, 1] < 0.5 * forms[:, 2]) | (forms[:, 1] > 2 * forms[:, 2])

    n = len(rec.U)
    run_panel(node, rec.bins[2], n)
    stop_panel(node)

    its = node["its"]
    steps = [i for i in its if i["trials"][-1]["accepted"]]
    every = [i for v in nodes.values() for i in v["its"]]
    last, before = its[-1], its[-2]
    numbers = {
        "code": format(code, "b"), "points": n, "halvings": c["halvings"], "armijo": c["armijo"],
        "rounding": c["rounding"], "saturation": c["saturation"], "soft_iters": plan.settings.soft_iters,
        "iterations": len(its), "steps": len(steps), "trials": sum(len(i["trials"]) for i in its),
        "first_trial_steps": sum(len(i["trials"]) == 1 for i in steps),
        "reason": node["reason"], "J_first": round(its[0]["J"], 3), "J_last": round(its[-1]["J"], 3),
        "hard_bins": round(rec.bins[2] / n, 3), "hard_final": round(node["sse_final"] / n, 3),
        "before_trial": len(before["trials"]) - 1,
        "last_step": f"{last['step']:.1e}", "last_margin": f"{last['margin']:.1e}",
        "last_best": f"{min(tr['dF'] for tr in last['trials']):.1e}",
        "gap_max": f"{max(i['gap'] for i in every):.0e}", "margin_min": f"{min(i['margin'] for i in every):.0e}",
        "fits": len(forms), "residual_form_error_percent": round(100 * float(res_err.max())),
        "sum_form_off": int(sum_off.sum()), "exact_gap_range": [f"{forms[:, 2].min():.0e}", f"{forms[:, 2].max():.0e}"],
        "nodes": len(nodes), "counters": {r: counted[r] for r in REASONS},
        "improved": plan.counters["soft_improved"],
        "iterations_all": sum(len(v["its"]) for v in nodes.values()),
        "trials_all": sum(len(i["trials"]) for i in every),
    }  # fmt: skip
    (mf.OUT / "apx-soft.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers))


if __name__ == "__main__":
    main()
