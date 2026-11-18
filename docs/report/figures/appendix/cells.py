"""Draw the panels of the companion's appendix on AFINNI's answer cells (apx-cells-*.svg, apx-cells.json).

Usage, from the repository root:

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python docs/report/figures/appendix/cells.py

The answering leaf nu(x) of every grid query comes from the library (``predict_local`` with
``return_cluster_assignments=True``). The script restates nu from the definition (the kappa
Euclidean candidates, then the smallest trust-region distance), the predictions with
``k_nn = 1`` and ``k_nn = 2``, ``exact_outside`` and ``strict_trust_region``, and stops unless
every restatement matches the library.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures as mf  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from scipy import ndimage  # noqa: E402

from src import IS_CUPY_BACKEND, to_host  # noqa: E402

GRID = 240  # the grid of the companion's error maps
MIN_PIECE = 10  # grid queries a separate piece of a cell must hold to be counted


def answers(
    s, X: np.ndarray, kappa: int, k_nn: int = 1, **kw
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(predictions, flag, answering leaves)`` from the library, for ``kappa = k_nn * coarse_factor`` candidates."""
    pred, out, nu = s.predict_local(X, k_nn=k_nn, coarse_factor=kappa // k_nn, return_cluster_assignments=True,
                                    strict_trust_region=kw.pop("strict", False), **kw)  # fmt: skip
    return (
        np.asarray(to_host(pred)),
        np.asarray(to_host(out)).astype(bool),
        np.asarray(to_host(nu)),
    )


def main() -> None:
    if IS_CUPY_BACKEND:
        raise SystemExit(
            "Run with SURROGATE_DEVICE=cpu: the panels and their numbers are the CPU results."
        )
    s, levels, _ = mf.record_afinni()
    leaves = [nd for nd in mf.afinni_forest(levels) if nd.leaf]
    C, T = to_host(s.X_src), to_host(s.trust_matrices_src)
    A, b = (to_host(s.model_params()[k]) for k in ("A_fit", "b_fit"))
    n = len(C)
    grid = mf.domain_grid(GRID)
    rows = np.arange(len(grid))

    # The definition: the kappa nearest centres (ties to the smaller index), then the smallest d_i.
    D = np.einsum(
        "qnd,nde,qne->qn", grid[:, None] - C[None], T, grid[:, None] - C[None]
    )
    euclid = np.argsort(
        np.linalg.norm(grid[:, None] - C[None], axis=2), axis=1, kind="stable"
    )
    best = D.argmin(axis=1)  # every leaf a candidate
    inside = D.min(axis=1) <= 1

    def restated(kappa: int) -> np.ndarray:
        cand = euclid[:, :kappa]
        return cand[rows, np.argmin(D[rows[:, None], cand], axis=1)]

    nu, share = {}, {}
    for kappa in [*range(1, mf.KAPPA + 1), n]:
        pred, out, idx = answers(s, grid, kappa)
        nu[kappa] = idx[:, 0]
        if not np.array_equal(nu[kappa], restated(kappa)):
            raise SystemExit(
                f"The restated answering leaf disagrees with predict at kappa = {kappa}."
            )
        if not np.allclose(
            pred[:, 0],
            np.einsum("qd,qd->q", A[nu[kappa], 0], grid) + b[nu[kappa], 0],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise SystemExit(
                "predict's answer is not the answering leaf's affine model."
            )
        if not np.array_equal(out, D[rows, nu[kappa]] > 1):
            raise SystemExit("predict's flag is not d_nu > 1.")
        share[kappa] = float(np.mean(nu[kappa] != best))
    if not np.array_equal(nu[n], best):
        raise SystemExit(
            "With every leaf a candidate, nu is not the global argmin of d_i."
        )
    pub_pred, pub_out = (
        np.asarray(to_host(v)) for v in s.predict(grid, k_nn=1, coarse_factor=mf.KAPPA)
    )
    if not np.array_equal(
        pub_pred, answers(s, grid, mf.KAPPA)[0]
    ) or not np.array_equal(pub_out.astype(bool), D[rows, nu[mf.KAPPA]] > 1):
        raise SystemExit("predict and predict_local disagree.")

    # exact_outside at one candidate: the flag becomes exact, the leaf only for the rows it flagged.
    k_low = 1
    _, out_x, idx_x = answers(s, grid, k_low, exact_outside=True)
    nu_x = idx_x[:, 0]
    flagged = D[rows, nu[k_low]] > 1
    expect = np.where(flagged & inside, best, nu[k_low])
    if not np.array_equal(out_x, ~inside) or not np.array_equal(nu_x, expect):
        raise SystemExit(
            "exact_outside does not keep unflagged rows and answer contained flagged rows by the global argmin."
        )

    # A sufficient condition for kappa candidates to find the global argmin: every other leaf j is at
    # Euclidean distance >= rho_kappa, the kappa-th nearest, so d_j >= lambda_* rho_kappa^2.
    lam = float(np.linalg.eigvalsh(T)[:, 0].min())
    rho2 = np.sort(np.linalg.norm(grid[:, None] - C[None], axis=2) ** 2, axis=1)[
        :, mf.KAPPA - 1
    ]
    certified = D[rows, nu[mf.KAPPA]] < lam * rho2
    if np.any(certified & (nu[mf.KAPPA] != best)):
        raise SystemExit(
            "The sufficient condition holds at a query whose answer is not the global argmin."
        )

    # The blend with k_nn = 2, restated, and strict_trust_region, which answers 0 where no leaf counts.
    pred2, _, idx2 = answers(s, grid, mf.KAPPA, k_nn=2)
    d2 = D[rows[:, None], idx2]
    w = 1.0 / (d2**2 + 1e-12)
    m2 = np.einsum("qkd,qd->qk", A[idx2, 0], grid) + b[idx2, 0]
    if not np.allclose(
        pred2[:, 0], (w * m2).sum(axis=1) / w.sum(axis=1), rtol=1e-12, atol=1e-12
    ):
        raise SystemExit("The restated blend disagrees with predict at k_nn = 2.")
    pred_s, out_s, _ = answers(s, grid, mf.KAPPA, strict=True)
    pred_1 = answers(s, grid, mf.KAPPA)[0]
    if np.any(np.isnan(pred_s)) or not np.array_equal(pred_s[out_s, 0], np.zeros(int(out_s.sum()))) or not np.array_equal(
            pred_s[~out_s], pred_1[~out_s]):  # fmt: skip
        raise SystemExit(
            "strict_trust_region does not answer 0 exactly at the flagged queries."
        )

    # The cells: pieces per leaf, and training points answered by another leaf.
    best_grid = best.reshape(GRID, GRID)
    # Pieces of at least MIN_PIECE queries: a lone query where two edges meet is the grid's, not the cell's.
    labelled = [
        ndimage.label(best_grid == i, structure=np.ones((3, 3)))[0] for i in range(n)
    ]
    pieces = [
        int(np.sum(np.bincount(lab.ravel())[1:] >= MIN_PIECE)) for lab in labelled
    ]
    own = [answers(s, nd.X, mf.KAPPA)[2][:, 0] for nd in leaves]
    foreign = sum(int(np.sum(o != i)) for i, o in enumerate(own))
    points = sum(len(nd.X) for nd in leaves)

    extent = (mf.LOWER[0], mf.UPPER[0], mf.LOWER[1], mf.UPPER[1])
    G0, G1 = grid[:, 0].reshape(GRID, GRID), grid[:, 1].reshape(GRID, GRID)
    tint = np.array([0.45 * np.array(to_rgb(nd.colour)) + 0.55 for nd in leaves])

    def cells(labels: np.ndarray, marked: np.ndarray | None):
        def draw(ax):
            lab = labels.reshape(GRID, GRID)
            ax.imshow(
                tint[lab],
                origin="lower",
                extent=extent,
                interpolation="nearest",
                zorder=1,
            )
            # Cell boundaries, so that two cells of one colour stay apart.
            for i in range(n):
                if np.any(lab == i):
                    ax.contour(
                        G0,
                        G1,
                        (lab == i).astype(float),
                        levels=[0.5],
                        colors="white",
                        linewidths=0.5,
                        zorder=2,
                    )
            for i, nd in enumerate(leaves):
                mf.ellipse(ax, C[i], T[i], nd.colour, lw=0.7, zorder=3)
            mf.cross(ax, C, [nd.colour for nd in leaves], size=14, lw=1.0, zorder=5)
            if marked is not None and marked.any():
                M = marked.reshape(GRID, GRID).astype(float)
                ax.contourf(
                    G0,
                    G1,
                    M,
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=["xxx"],
                    zorder=4,
                )
                ax.contour(
                    G0, G1, M, levels=[0.5], colors=mf.DARK, linewidths=0.5, zorder=4
                )

        return draw

    mf.panel("apx-cells-all", cells(best, None), size=(2.1, 2.1), back=False)
    mf.panel(
        "apx-cells-one",
        cells(nu[k_low], nu[k_low] != best),
        size=(2.1, 2.1),
        back=False,
    )
    mf.panel("apx-cells-exact", cells(nu_x, nu_x != best), size=(2.1, 2.1), back=False)

    first_zero = next(
        k
        for k in range(1, mf.KAPPA + 1)
        if all(share[j] == 0 for j in range(k, mf.KAPPA + 1))
    )
    pct = lambda v: round(100 * v, 1)  # noqa: E731
    numbers = {
        "grid": int(len(grid)), "leaves": int(n), "kappa": mf.KAPPA, "k_low": k_low,
        "share": {str(k): pct(share[k]) for k in range(1, mf.KAPPA + 1)},
        "first_zero": int(first_zero),
        "exact_share": pct(np.mean(nu_x != best)),
        "flagged_low": pct(flagged.mean()), "flag_fixed": pct(np.mean(flagged & inside)),
        "certified": pct(certified.mean()), "lambda_min": round(lam, 2),
        "pieces_max": int(max(pieces)), "min_piece": MIN_PIECE, "split_cells": int(sum(p > 1 for p in pieces)),
        "foreign_points": foreign, "points": int(points), "foreign_percent": pct(foreign / points),
        "strict_zero": int(out_s.sum()), "strict_zero_percent": pct(out_s.mean()),
    }  # fmt: skip
    (mf.OUT / "apx-cells.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers))


if __name__ == "__main__":
    main()
