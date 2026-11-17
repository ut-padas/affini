"""
The run matrix of Part B: the reference ART and this library's ART on the reference's benchmark
problems, as one ``scripts/art_comparison.py`` call per (case, side).

Prints tab-separated lines ``case_id  group  tier  side  nprocs  device  out_json  args...``:
``side`` is ``reference`` (the reference ART) or ``library`` (this library's ART), ``device`` is ``main`` (the
sweep's device) or ``cpu`` (forced, for the CPU half of the GPU-vs-CPU check), ``out_json`` is
relative to the results directory and ``args`` go to art_comparison.py after ``--``.

Matched settings (group ``conv``): same target, domain, test set (``10000 d`` uniform points,
seed 19), seed, leaf degree, pointwise relative tolerance (floor 1e-12), ``--max-depth``, sample
multiplier 50 and soft-oblique split search with the same iteration cap (``--library-soft-iters
200``, the reference's ``--max-iters``, pinned so the match does not depend on the default). Group
``library_default`` reruns this library's side with its own default (``ARTSettings.soft_iters``). Problems follow the reference's recorded cluster presets
(``tasmanian/oracle_presets.json``): offsets keep |f| away from 0, where a pointwise relative
tolerance cannot be met by any leaf.

Tiers bound the expected cost: S (seconds to minutes on a CPU), M (up to about an hour), L
(hours; deep trees). Usage: ``art_cases.py [--smoke] [--groups ...] [--max-tier S|M|L]
[--seeds 1,2,3] [--nps 1,2,4] [--gpu] [--only REGEX]``.
"""

import argparse
import math
import re

GROUPS = ("conv", "library_default", "depth", "exact_target", "mpi", "repeat", "gpu_vs_cpu")
TIERS = {"S": 0, "M": 1, "L": 2}
SIGNS = [1, -1, 1, 1, -1, -1, 1, -1]  # the reference's recorded dense-sign plane-wave normals (d = 3, 5, 8)
RASTRIGIN = {3: (5.12, 22), 4: (2.5, 21), 5: (1.5, 20)}  # half-width, max depth of the recorded runs

# name: (benchmark arguments for dimension d, dimensions, leaf degrees, tolerances, max depth)
PROBLEMS = {
    "quadratic": (lambda d: ["--benchmark", "quadratic", "--offset", "1"], (2, 3, 4, 5), (1,),
                  (0.1, 0.05, 0.02, 0.01, 0.005), 20),
    "gaussian": (lambda d: ["--benchmark", "gaussian"], (2, 3, 4, 5), (1, 2), (0.1, 0.05, 0.02, 0.01, 0.005), 20),
    "gaussian_mixture": (lambda d: ["--benchmark", "gaussian_mixture"], (2, 3), (1,), (0.1, 0.05, 0.02, 0.01), 20),
    "plane_wave": (lambda d: ["--benchmark", "plane_wave", "--plane-wave-frequency", "4", "--offset", "2",
                              "--plane-wave-normal", *map(str, SIGNS[:d])], (3, 5, 8), (1,),
                   (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001), 24),
    "rosenbrock": (lambda d: ["--benchmark", "rosenbrock", "--offset", "1"], (2, 3, 4, 5), (1, 2),
                   (0.1, 0.05, 0.02, 0.01), 20),
    "rastrigin": (lambda d: ["--benchmark", "rastrigin", "--offset", "1", "--domain-low", str(-RASTRIGIN[d][0]),
                             "--domain-high", str(RASTRIGIN[d][0])], (3, 4, 5), (2,), (0.1, 0.05, 0.02), None),
}  # fmt: skip
SPHERE_DEPTHS = (6, 8, 10, 12, 14, 16)  # spherical_piecewise: discontinuous, so a depth sweep at tol 0.01


def tier(problem: str, d: int, degree: int, tol: float, depth: int) -> str:
    """S / M / L from the expected leaf count ~ tol^(-d_eff/(degree+1)); rank-1 plane waves count as d_eff = 1."""
    if problem == "rastrigin":
        return "M" if d == 3 else "L"
    d_eff = 1 if problem == "plane_wave" else d
    score = d_eff / (degree + 1) * math.log10(0.1 / tol) + (0.75 if d >= 8 else 0.0)
    return "S" if score <= 1.5 else "M" if score <= 2.6 else "L"


def args_for(problem, d, degree, tol, depth, seed):
    base = PROBLEMS[problem][0](d) if problem in PROBLEMS else ["--benchmark", problem]
    return [*base, "--dimension", str(d), "--leaf-degree", str(degree), "--error-tolerance", str(tol),
            "--max-depth", str(depth), "--seed", str(seed)]  # fmt: skip


def full_matrix(seeds, nps, gpu):
    rows = []

    def both(case, group, t, args, sides=("reference", "library"), np_=1, device="main", tag=""):
        for side in sides:
            extra = ["--library-soft-iters", "200"] if group != "library_default" else []
            rows.append((case, group, t, side, np_, device, f"{group}/{case}.{side}{tag}.json", args + extra))

    for problem, (_, dims, degrees, tols, depth) in PROBLEMS.items():
        for d in dims:
            for degree in degrees:
                D = RASTRIGIN[d][1] if problem == "rastrigin" else depth
                for tol in tols:
                    t = tier(problem, d, degree, tol, D)
                    for seed in seeds:
                        case = f"{problem}_d{d}_deg{degree}_tol{tol}_s{seed}"
                        both(case, "conv", t, args_for(problem, d, degree, tol, D, seed))
                    both(f"{problem}_d{d}_deg{degree}_tol{tol}_s{seeds[0]}", "library_default", t,
                         args_for(problem, d, degree, tol, D, seeds[0]), sides=("library",))  # fmt: skip
    for d in (2, 3, 4, 5):
        for depth in SPHERE_DEPTHS:
            t = "S" if depth <= 10 else "M" if depth <= 14 and d <= 3 else "L"
            for seed in seeds:
                both(f"spherical_piecewise_d{d}_deg1_depth{depth}_s{seed}", "depth", t,
                     args_for("spherical_piecewise", d, 1, 0.01, depth, seed))  # fmt: skip
    # Exactly representable target: quadratic leaves on the quadratic must stop at the root in both codes.
    for d in (2, 3, 5):
        both(f"quadratic_d{d}_deg2_exact", "exact_target", "S",
             args_for("quadratic", d, 2, 1e-8, 6, seeds[0]))  # fmt: skip
    # This library's ART promises the same tree for any rank count and steal schedule (docs/USAGE.md).
    mpi_case = args_for("rosenbrock", 3, 1, 0.02, 20, seeds[0])
    for p in nps:
        rows.append(("rosenbrock_d3_deg1_tol0.02", "mpi", "S", "library", p, "main",
                     f"mpi/rosenbrock_d3_deg1_tol0.02.library.np{p}.json",
                     mpi_case + ["--library-soft-iters", "200"] + (["--library-work-stealing"] if p > 1 else [])))  # fmt: skip
    # Both codes must reproduce their own run (same seed, same machine): the baseline of "same".
    for case, args in (("rosenbrock_d3_deg1_tol0.02", mpi_case), ("plane_wave_d5_deg1_tol0.01",
                                                                     args_for("plane_wave", 5, 1, 0.01, 24, seeds[0]))):
        both(case, "repeat", "S", args, tag=".repeat")
    if gpu:  # the same runs of this library's ART on the CPU, to compare with the sweep's GPU runs
        for problem, d, degree, tol in (("rosenbrock", 3, 1, 0.02), ("gaussian", 4, 1, 0.01), ("plane_wave", 5, 1, 0.005)):
            case = f"{problem}_d{d}_deg{degree}_tol{tol}_s{seeds[0]}"
            D = PROBLEMS[problem][4]
            rows.append((case, "gpu_vs_cpu", "S", "library", 1, "cpu", f"gpu_vs_cpu/{case}.library.cpu.json",
                         args_for(problem, d, degree, tol, D, seeds[0]) + ["--library-soft-iters", "200"]))  # fmt: skip
    return rows


def smoke_matrix(nps):
    rows = []
    small = ["--n-test", "2000"]
    for problem in PROBLEMS:
        d = 3 if problem in ("rastrigin", "plane_wave") else 2
        degree = 2 if problem == "rastrigin" else 1
        for side in ("reference", "library"):
            rows.append((f"{problem}_d{d}_smoke", "conv", "S", side, 1, "main", f"conv/{problem}_d{d}_smoke.{side}.json",
                         args_for(problem, d, degree, 0.1, 4, 1) + small + ["--library-soft-iters", "200"]))  # fmt: skip
    for side in ("reference", "library"):
        rows.append(("spherical_piecewise_d2_smoke", "depth", "S", side, 1, "main",
                     f"depth/spherical_piecewise_d2_smoke.{side}.json",
                     args_for("spherical_piecewise", 2, 1, 0.01, 4, 1) + small))  # fmt: skip
        rows.append(("quadratic_d2_deg2_exact", "exact_target", "S", side, 1, "main",
                     f"exact_target/quadratic_d2_deg2_exact.{side}.json", args_for("quadratic", 2, 2, 1e-8, 3, 1) + small))  # fmt: skip
    for p in nps:
        rows.append(("rosenbrock_d2_smoke", "mpi", "S", "library", p, "main", f"mpi/rosenbrock_d2_smoke.library.np{p}.json",
                     args_for("rosenbrock", 2, 1, 0.05, 6, 1) + small + (["--library-work-stealing"] if p > 1 else [])))  # fmt: skip
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--groups", default=",".join(GROUPS))
    parser.add_argument("--max-tier", default="L", choices=tuple(TIERS))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--nps", default="1,2,4")
    parser.add_argument("--gpu", action="store_true", help="the sweep runs this library's side on a GPU: add the CPU twins")
    parser.add_argument("--only", default=None, help="regular expression; keep the case ids it matches")
    a = parser.parse_args()
    groups = set(a.groups.split(","))
    if groups - set(GROUPS):
        parser.error(f"unknown groups {sorted(groups - set(GROUPS))}; choose from {GROUPS}")
    nps = [int(p) for p in a.nps.split(",")]
    rows = smoke_matrix(nps[:2]) if a.smoke else full_matrix([int(s) for s in a.seeds.split(",")], nps, a.gpu)
    for case, group, t, side, p, device, out, args in rows:
        if group in groups and TIERS[t] <= TIERS[a.max_tier] and (a.only is None or re.search(a.only, case)):
            print("\t".join([case, group, t, side, str(p), device, out, *args]))


if __name__ == "__main__":
    main()
