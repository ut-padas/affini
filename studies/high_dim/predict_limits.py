"""
Predicted memory and work per node for every variant and d, from the plans' own batch charge
(``bytes_per_point``, instantiated on the CPU exactly as a run builds them) and from a
term-by-term estimate of what the code allocates, for comparison with the study's
measurements. Optionally (``--root-fit``) the error of one global least-squares fit, which
predicts the coarsest tolerance at which a single node is accepted.

    AFINNI_DEVICE=cpu AFINNI_MPI=off .venv/bin/python -m studies.high_dim.predict_limits \
        --out studies/high_dim/results/predicted.csv

Columns (bytes unless noted):

- ``charged_per_point``: ``plan.bytes_per_point()``; ``charged_per_node`` times the points a node
  occupies (``plan.work_sizes``; AFINNI: an augmented rejected cluster, 2 min_pts_per_cluster).
- ``cap_points_<G>GB``: the batch cap ``_cap_batch_size`` computes on a card of G GB
  (65% of free memory, free = G - 0.6 GB for the CUDA context), and ``nodes_per_batch_<G>GB``.
  Below one node, a batch holds one node alone and may exceed the budget.
- ``est_per_node``: an estimate of the node's real peak from reading the code (ART: point
  arrays, the 256-row partial sums of ``segment_cross``, the bins search's Gram stacks
  4 (n_bins - 1) p^2 at the solve, the soft search's 6 p^2; AFINNI: the charge plus about eight
  d x d arrays per cluster, which the charge omits). ``fits_<G>GB`` compares it with the free
  memory. These are reading estimates, not measurements: the study's ``gpu_pool_peak_bytes``
  is the measured value.
- ``leaf_row``: host bytes per leaf; ``leaf_buffer_reserve``: the up-front leaf buffer at the
  library default ``max_clusters_est`` = 1,000,000 (the study uses 20,000).
- ``knn_gather_chunk``: AFINNI's prediction gather for a default chunk (5000 queries x 10
  candidates x d^2 doubles).
- ``madds_per_node``: multiply-adds of one ART node's fit and split search (Grams n p^2 per
  pass: one fit, n_directions bins passes, 3 screened re-fits, soft_iters soft iterations of
  two weighted Grams; plus the Cholesky solves), and ``s_per_node_at_1TF`` at an effective 1e12
  fp64 multiply-adds per second (consumer cards run fp64 at 1/32 to 1/64 of fp32; calibrate
  with the measured time per node).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

CARDS_GB = (24, 48, 80)
CONTEXT_GB = 0.6
DIMS = (10, 20, 40, 60, 80, 100, 120, 160, 200, 250, 300)


def plan_for(variant, d: int, fd: int = 1):
    """The plan a run of ``variant`` builds at dimension d (on the CPU), and its config."""
    from src.afinni import AFINNIPlan
    from src.art import ARTPlan, Domain, Roots
    from studies.high_dim.problems import domain
    from studies.high_dim.settings import build_config

    config = build_config(variant, d, atol=1e-2, profile=False)
    if variant.algorithm == "afinni":
        return AFINNIPlan(config, d, fd, 0), config
    lower, upper = domain(d)
    return ARTPlan(config, d, fd, Domain(lower, upper), Roots.build("box", d, 1)), config


def estimate(variant, d: int, fd: int = 1) -> dict:
    """One row of the prediction table (see the module docstring)."""
    from studies.high_dim.settings import min_pts, n_coefficients

    plan, config = plan_for(variant, d, fd)
    bpp = int(plan.bytes_per_point())
    p = n_coefficients(variant.leaf, d)
    h = config.max_levels + 1
    if variant.algorithm == "art":
        n = int(plan.work_sizes(np.zeros(1, np.int64))[0])
        s = config.art
        nb, D = s.n_bins, s.n_directions
        partial = n * (p * (p + 1) // 2) / 256  # segment_cross: one partial Gram per 256-row chunk
        points = n * (5 * d + 3 * fd + 2 * p + 4)
        bins = 4 * (nb - 1) * (p * p + p * fd) + partial
        soft = (6 * p * p + 4 * p * fd + 2 * partial) if s.split_search == "soft" else 0
        est = 8 * (points + max(bins, soft) + 3 * h * (d + 1) + D * d)
        soft_iters = s.soft_iters if s.split_search == "soft" else 0
        madds = n * p * p * (1 + D + s.n_screened * 2 + 2 * soft_iters) + (2 * D * (nb - 1) + 4 * soft_iters) * p**3 / 3
        leaf_row = 8 * (d + 3 * fd + fd * p + 1) + 4 + 8 * (d + 3)  # leaf + its split record
    else:
        n = 2 * min_pts(d)
        est = n * bpp + 8 * 8 * d * d
        madds = n * d * d * 6 + 8 * d**3  # stats, fits, Hessian estimate; eigh ~ 8 d^3 / 3 per call
        leaf_row = 8 * (d * d + d + 4 * fd + fd * d + 1)
    row = {
        "variant": variant.name, "algorithm": variant.algorithm, "leaf": variant.leaf, "d": d, "p": p,
        "points_per_node": n, "charged_per_point": bpp, "charged_per_node": n * bpp, "est_per_node": int(est),
        "est_over_charged": est / max(1, n * bpp), "leaf_row": leaf_row,
        "leaf_buffer_reserve": leaf_row * 1_000_000, "knn_gather_chunk": 8 * 5000 * 10 * d * d,
        "madds_per_node": float(madds), "s_per_node_at_1TF": float(madds) / 1e12,
        "f_evals_per_node": n,
    }  # fmt: skip
    for g in CARDS_GB:
        free = (g - CONTEXT_GB) * 1e9
        cap = int(np.clip(int(free * 0.65) // max(1, bpp), plan.batch_floor, 20_000_000))
        row[f"cap_points_{g}GB"] = cap
        row[f"nodes_per_batch_{g}GB"] = cap / n
        row[f"fits_{g}GB"] = bool(est <= free)
    return row


def root_fit(problem, d: int, leaf: str, m: float = 50.0, n_test: int = 20_000) -> dict:
    """Error of one least-squares fit over the whole box, relative to the problem's scale: the coarsest tolerance a single node meets."""
    from studies.high_dim.problems import scoring_set, uniform_points
    from studies.high_dim.settings import n_coefficients

    p = n_coefficients(leaf, d)
    n = int(np.ceil(m * p))
    X = uniform_points(n, d, seed=777 + d)
    F = np.asarray(problem.f(X))[:, 0]

    def features(Z):
        if leaf == "affine":
            return np.c_[np.ones(len(Z)), Z]
        i, j = np.triu_indices(d)
        return np.c_[np.ones(len(Z)), Z, Z[:, i] * Z[:, j]]

    beta = np.linalg.lstsq(features(X), F, rcond=None)[0]
    Xq, Fq, scale = scoring_set(problem, d, n_test)
    train_err = np.abs(features(X) @ beta - F)
    test_err = np.abs(features(Xq) @ beta - Fq[:, 0])
    return {
        "problem": problem.name, "leaf": leaf, "d": d, "n": n, "scale": scale,
        "sample_max_rel": float(train_err.max() / scale),  # what the Linf acceptance test sees
        "test_max_rel": float(test_err.max() / scale), "test_rms_rel": float(np.sqrt(np.mean(test_err**2)) / scale),
    }  # fmt: skip


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def gb(x: float) -> str:
    return f"{x / 1e9:.3g}"


def main(argv=None) -> int:
    from studies.high_dim.problems import PROBLEMS
    from studies.high_dim.settings import VARIANTS

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dims", default=",".join(map(str, DIMS)))
    parser.add_argument("--variants", default="afinni,art_affine,art_quad,art_affine_m10,art_quad_m4")
    parser.add_argument("--out", default=None, help="CSV path for the memory table")
    parser.add_argument("--root-fit", action="store_true", help="also fit one global model per problem and d")
    parser.add_argument("--root-fit-max-p", type=int, default=1000, help="skip root fits with more coefficients")
    parser.add_argument("--root-fit-out", default=None)
    args = parser.parse_args(argv)
    dims = [int(x) for x in args.dims.split(",")]
    rows = [estimate(VARIANTS[v], d) for v in args.variants.split(",") for d in dims]
    if args.out:
        write_csv(rows, Path(args.out))
    print("| variant | d | p | n/node | charged B/pt | charged/node GB | est/node GB | est/charged | nodes/batch 24 GB | "
          "fits 24/48/80 GB | leaf row KB | 1M-leaf reserve GB | s/node at 1 TF |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        fits = "/".join("y" if r[f"fits_{g}GB"] else "n" for g in CARDS_GB)
        print(f"| {r['variant']} | {r['d']} | {r['p']} | {r['points_per_node']:,} | {r['charged_per_point']:,} | "
              f"{gb(r['charged_per_node'])} | {gb(r['est_per_node'])} | {r['est_over_charged']:.2f} | "
              f"{r['nodes_per_batch_24GB']:.3g} | {fits} | {r['leaf_row'] / 1e3:.3g} | {gb(r['leaf_buffer_reserve'])} | "
              f"{r['s_per_node_at_1TF']:.3g} |")  # fmt: skip
    if args.root_fit:
        fits = []
        from studies.high_dim.settings import n_coefficients

        for name, problem in PROBLEMS.items():
            for leaf in ("affine", "quadratic"):
                for d in dims:
                    if n_coefficients(leaf, d) > args.root_fit_max_p:
                        continue
                    fits.append(root_fit(problem, d, leaf))
                    print(f"root fit {name:16s} {leaf:9s} d={d:3d}: sample max {fits[-1]['sample_max_rel']:.3g}, "
                          f"test max {fits[-1]['test_max_rel']:.3g}, test rms {fits[-1]['test_rms_rel']:.3g} (x scale)",
                          flush=True)  # fmt: skip
        if args.root_fit_out:
            write_csv(fits, Path(args.root_fit_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
