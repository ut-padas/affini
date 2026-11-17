"""
The run matrix that re-runs the AFINNI manuscript's experiments, one launcher call per line.

Prints tab-separated lines ``case_id  group  tier  nprocs  seed  kind  args...`` that
`run_paper.sh` executes: ``kind`` is ``launcher`` (args go to ``afinni_launcher.py`` through
`run_case.py`) or ``iterative`` (args go to `run_iterative.py`). Settings follow
``run_paper_experiments.sh`` where it has the experiment, and the manuscript's text where it does
not (Table 2's cluster sizes, Table 3, the e_nh column, the iterative-refinement figure,
MPI). Tiers bound the cost by the paper's evaluation count N: S <= 1e7 (minutes on one GPU,
a few GB), M <= 1e8 (up to about an hour, tens of GB of host RAM), L above (an HPC node with
>= 128 GB of host RAM; the paper used a GH200 with 480 GB).

Usage: ``cases.py [--smoke] [--groups g1,g2] [--max-tier S|M|L] [--seeds 0,1,2] [--nps 1,2,4,8] [--only REGEX]``.
"""

import argparse
import re

from paper_values import _CONVERGENCE, _LORENZ_DIMS, _ROSEN_WEAK, _WAVE_EXACT

GROUPS = ("conv", "conv_nh", "conv_k10", "lorenz_dims", "rosen_weak", "wave_exact", "iterative", "mpi", "seeds")
TIERS = {"S": 0, "M": 1, "L": 2}
EXPERIMENT = {
    "anisotropic_quadratic_2": "anisotropic_quadratic", "anisotropic_gaussian": "anisotropic_gaussian",
    "rosenbrock": "rosenbrock", "lorenz96_dt_0.1": "lorenz96", "wave": "wave",
}  # fmt: skip
# --coarse_factor (kappa candidates) per d for Table 1: max(50, 2 d^2), as run_paper_experiments.sh
# passes it (50 / 50 / 200). The paper-era code (91ddd97) reproduces Wave rows #38-39 (N, N_c, p_o,
# e_in, e) to the printed digits with 50 at d = 2, and with 10 gives 24-42x the printed p_o
# (studies/sanity/README.md, "Where the differences come from"). Group conv_k10 reruns the d = 2 rows
# with 10. Table 2 (Lorenz-96 up to d = 80) keeps the launcher's 10.
def coarse_factor(d: int) -> int:
    return max(50, 2 * d * d)


COARSE = {d: coarse_factor(d) for d in (2, 5, 10)}


def group_coarse_factor(group: str, d: int) -> int:
    """
    The coarse_factor a group's cases run with, which collect_paper.py also matches runs by: the
    launcher's 10 for Table 2, COARSE[d] for everything else (Figure 8's exact-gradient runs too,
    since its LSQ column is Table 1b's Wave d = 5 runs).
    """
    return 10 if group == "lorenz_dims" else COARSE[d]


def lorenz_dims_sizes(d: int) -> tuple[int, int, int]:
    """
    Table 2's (N_0, K_0, M) at dimension d, as the code that produced the table set them
    (91ddd97, cap_initial_augmentation_budget with its 16-cluster floor): N_0 = min(5e4 1.3^d, 1e6),
    then M = 100 1.3^d is lowered until 16 clusters of 2 M points fit in N_0, and K_0 = 25 1.3^d
    until K_0 clusters do. These reproduce the table's N_c within 5% at d = 10, 20 and 40; the
    caption's K_0 = min(25 1.3^d, 1000), M = min(100 1.3^d, 500) gives about 77x the leaves and
    87% p_o at d = 40.
    """
    n0 = min(int(5e4 * 1.3**d), 10**6)
    m = min(int(100 * 1.3**d), max(2, n0 // (2 * 16)))
    k0 = min(int(25 * 1.3**d), max(1, n0 // (2 * m)))
    return n0, k0, m


def tier(evaluations: float) -> str:
    return "S" if evaluations <= 1e7 else "M" if evaluations <= 1e8 else "L"


def launcher_args(problem: str, d: int, mode: str, coarse: int, tols=None, extra=()) -> list[str]:
    args = ["--experiment", EXPERIMENT[problem], "--dims", str(d), "--error_eval_mode", mode, "--coarse_factor", str(coarse)]
    if tols is not None:
        args += ["--tols", *map(str, tols)]
    if problem.startswith("lorenz96"):
        args += ["--dt", "0.1"]  # the launcher's default, stated so a changed default cannot move the problem
    return args + list(extra)


def full_matrix(seeds: list[int], nps: list[int]) -> list[tuple]:
    """(case_id, group, tier, nprocs, seed, kind, args) for every run of the full study."""
    rows = []
    for _row, problem, d, tol, N, *_ in _CONVERGENCE:
        name = f"{EXPERIMENT[problem]}_d{d}_tol{tol}"
        rows.append((f"conv_{name}", "conv", tier(N), 1, 0, "launcher",
                     launcher_args(problem, d, "affine_cov_hess", COARSE[d], [tol])))  # fmt: skip
        # e_nh: the same run without Hessian-informed splits; its N is not reported, so the tier is the
        # Hessian run's (expect more evaluations).
        rows.append((f"conv_nh_{name}", "conv_nh", tier(N), 1, 0, "launcher",
                     launcher_args(problem, d, "affine_cov_no_hess", COARSE[d], [tol])))  # fmt: skip
        if d == 2:
            rows.append((f"conv_k10_{name}", "conv_k10", tier(N), 1, 0, "launcher",
                         launcher_args(problem, d, "affine_cov_hess", 10, [tol])))  # fmt: skip
        if tier(N) == "S":
            for s in seeds:
                if s != 0:
                    rows.append((f"seed{s}_{name}", "seeds", "S", 1, s, "launcher",
                                 launcher_args(problem, d, "affine_cov_hess", COARSE[d], [tol])))  # fmt: skip
    for _row, d, N, *_ in _LORENZ_DIMS:
        n0, k0, m = lorenz_dims_sizes(d)
        rows.append((f"lorenz_dims_d{d}", "lorenz_dims", tier(N), 1, 0, "launcher", launcher_args(
            "lorenz96_dt_0.1", d, "affine_cov_hess", group_coarse_factor("lorenz_dims", d), [0.1],
            ["--N_train", str(n0), "--num_initial_clusters", str(k0), "--min_pts_per_cluster", str(m)])))  # fmt: skip
    for atol, n0, nc, *_ in _ROSEN_WEAK:
        # Absolute test only (rtol off); N is not reported, about 500 evaluations per leaf in Table 1b.
        rows.append((f"rosen_weak_atol{atol}", "rosen_weak", tier(500 * nc), 1, 0, "launcher", launcher_args(
            "rosenbrock", 5, "affine_cov_hess", 50, None,
            ["--atols", str(atol), "--rtols", "none", "--N_train", str(n0)])))  # fmt: skip
    for tol, _lsq, (nc, *_rest) in _WAVE_EXACT:
        # Figure 8 sets these beside its LSQ column, which is Table 1b's Wave d = 5 runs, so they get
        # the same coarse_factor (50). run_paper_experiments.sh ran them with the launcher's 10, at
        # which the candidates miss the containing leaf of most flagged points (p_o 5-43% against an
        # exact 0.2-1.2%); training does not depend on it.
        rows.append((f"wave_exact_tol{tol}", "wave_exact", tier(370 * nc), 1, 0, "launcher",
                     launcher_args("wave", 5, "affine_cov_hess_exact_grad", group_coarse_factor("wave_exact", 5), [tol])))  # fmt: skip
    for fn in ("gaussian", "quadratic", "rosenbrock", "wave"):
        rows.append((f"iterative_{fn}_d2", "iterative", "S", 1, 0, "iterative",
                     ["--function", fn, "--dimension", "2", "--tol", "0.05", "--n-iter", "10"]))  # fmt: skip
    # MPI: Table 6's problem (Rosenbrock d=5, N_0 = 1e5) on several rank counts. AFINNI's trees depend on
    # the rank count and steal schedule, so this checks that accuracy and counts stay within noise.
    for tol, t in ((0.05, "M"), (0.005, "L")):
        for p in nps:
            rows.append((f"mpi_np{p}_tol{tol}", "mpi", t, p, 0, "launcher", launcher_args(
                "rosenbrock", 5, "affine_cov_hess", 50, [tol], ["--N_train", "100000"])))  # fmt: skip
    return rows


def smoke_matrix(nps: list[int]) -> list[tuple]:
    """One tiny run per group and code path (seconds each on a CPU)."""
    tiny = ["--N_train", "3000", "--num_initial_clusters", "8", "--min_pts_per_cluster", "40"]
    rows = []
    for problem in EXPERIMENT:
        d = 5 if problem.startswith("lorenz96") else 2
        name = f"{EXPERIMENT[problem]}_d{d}_tol0.1"
        rows.append((f"conv_{name}", "conv", "S", 1, 0, "launcher",
                     launcher_args(problem, d, "affine_cov_hess", 10, [0.1], tiny)))  # fmt: skip
    rows.append(("conv_nh_rosenbrock_d2_tol0.1", "conv_nh", "S", 1, 0, "launcher",
                 launcher_args("rosenbrock", 2, "affine_cov_no_hess", 10, [0.1], tiny)))  # fmt: skip
    rows.append(("seed1_rosenbrock_d2_tol0.1", "seeds", "S", 1, 1, "launcher",
                 launcher_args("rosenbrock", 2, "affine_cov_hess", 10, [0.1], tiny)))  # fmt: skip
    rows.append(("lorenz_dims_d10", "lorenz_dims", "S", 1, 0, "launcher", launcher_args(
        "lorenz96_dt_0.1", 10, "affine_cov_hess", 10, [0.1],
        ["--N_train", "4000", "--num_initial_clusters", "8", "--min_pts_per_cluster", "60"])))  # fmt: skip
    rows.append(("rosen_weak_atol579.8", "rosen_weak", "S", 1, 0, "launcher", launcher_args(
        "rosenbrock", 2, "affine_cov_hess", 10, None, ["--atols", "579.8", "--rtols", "none", *tiny])))  # fmt: skip
    rows.append(("wave_exact_tol0.1", "wave_exact", "S", 1, 0, "launcher",
                 launcher_args("wave", 2, "affine_cov_hess_exact_grad", 10, [0.1], tiny)))  # fmt: skip
    rows.append(("iterative_rosenbrock_d2", "iterative", "S", 1, 0, "iterative",
                 ["--function", "rosenbrock", "--dimension", "2", "--tol", "0.1", "--n-iter", "1"]))  # fmt: skip
    for p in nps:
        rows.append((f"mpi_np{p}_tol0.1", "mpi", "S", p, 0, "launcher",
                     launcher_args("rosenbrock", 2, "affine_cov_hess", 10, [0.1], tiny)))  # fmt: skip
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--max-tier", default="L", choices=tuple(TIERS))
    parser.add_argument("--seeds", default="0,1,2", help="data and training seeds of the 'seeds' group (0 is the paper's)")
    parser.add_argument("--nps", default="1,2,4,8", help="rank counts of the 'mpi' group")
    parser.add_argument("--only", default=None, help="regular expression; keep the case ids it matches")
    a = parser.parse_args()
    groups = set(a.groups.split(","))
    unknown = groups - set(GROUPS)
    if unknown:
        parser.error(f"unknown groups {sorted(unknown)}; choose from {GROUPS}")
    nps = [int(p) for p in a.nps.split(",")]
    rows = smoke_matrix(nps[:2]) if a.smoke else full_matrix([int(s) for s in a.seeds.split(",")], nps)
    for case_id, group, t, p, seed, kind, args in rows:
        if group in groups and TIERS[t] <= TIERS[a.max_tier] and (a.only is None or re.search(a.only, case_id)):
            print("\t".join([case_id, group, t, str(p), str(seed), kind, *args]))


if __name__ == "__main__":
    main()
