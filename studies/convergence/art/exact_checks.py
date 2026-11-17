"""
Exact-agreement check of the two ART codes where one is meaningful: the root fit.

Below the root the trees cannot be compared bit for bit: the codes draw their samples from
different generators and search splits differently. At the root both fit the same model class
to points in the same box, so feeding the reference's own root sample (``--store-samples``,
``--max-depth 0``) to this library's ART (``max_levels=0``, exactly ``ceil(m p)`` points, so no top-up)
must give the same leaf model to rounding, the same training error and the same accept/reject
decision. A NumPy least-squares fit in the reference's basis is the third opinion.

A second check, ``slab``, fits an exactly affine target on one thin box (aspect ratio 1 to
1e4) with both codes' leaf models. Both should reproduce it to rounding: this library's ART fits
without a ridge by default, as the reference does (``ARTSettings.leaf_fit="lstsq"``, the
reference's ``--ridge 0`` with its ``auto`` solver). With ``leaf_fit="tikhonov"`` (``affine_lsq``:
``1e-8 * trace`` of the scatter matrix; ``quadratic_lsq``: ``1e-10`` of the Gram's mean
diagonal) the error grows like ``1e-8 * aspect^2``, and deep ART nodes are thin.

Usage: ``exact_checks.py --art-path P --out-json F [--smoke]``; prints one line per case and
exits non-zero when a case fails. Root pass criteria: max |prediction difference| on 10,000
uniform test points <= 1e-6 * max |prediction| (room for ``leaf_fit="tikhonov"``, whose ridges bias
the fit by about 1e-8 on a cube), both codes' training max pointwise relative errors equal to 1e-6 relative, the same
decision unless the error is within 1e-6 of the tolerance, and this library's ART evaluating f only
at the given points (plus at most one centroid). Slab pass criterion: max pointwise relative error <= 1e-6
for each code separately; a failure there is reported per code.
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

# problem name -> reference benchmark arguments (the study's problems, CONVERGENCE_STUDY.md Part B)
PROBLEMS = {
    "quadratic": ["--benchmark", "quadratic", "--offset", "1"],
    "gaussian": ["--benchmark", "gaussian"],
    "rosenbrock": ["--benchmark", "rosenbrock", "--offset", "1"],
    "plane_wave": ["--benchmark", "plane_wave", "--plane-wave-frequency", "4", "--offset", "2"],
    "spherical_piecewise": ["--benchmark", "spherical_piecewise"],
    "rastrigin": ["--benchmark", "rastrigin", "--offset", "1"],
}


def cases(smoke: bool):
    """(problem, dimension, leaf degree, tolerance)."""
    if smoke:
        return [("rosenbrock", 2, 1, 0.05), ("quadratic", 2, 2, 1e-6)]
    out = []
    for problem in PROBLEMS:
        for d in (2, 3, 5):
            if problem == "spherical_piecewise" and d > 5:
                continue
            for degree in (1, 2):
                out.append((problem, d, degree, 0.05))
    out.append(("quadratic", 3, 2, 1e-6))  # exactly representable: both must accept the root
    return out


def plane_normal(d: int) -> list[str]:
    signs = [1, -1, 1, 1, -1, -1, 1, -1]  # the reference's recorded dense-sign directions (3, 5, 8)
    return ["--plane-wave-normal", *map(str, (signs * 2)[:d])]


def run_case(problem, d, degree, tol, art_path: Path) -> dict:
    from art.domain import BoxDomain
    from examples.tree_benchmark_helpers import build_tree_benchmark
    from examples.tree_high_dim_benchmark import BENCHMARK_DEFAULT_INTERVALS, make_benchmark_target, parse_args, validate_args

    from src import to_host, xp
    from src.config import ARTSettings, TrainingConfig
    from src.core.evaluation import Evaluator
    from src.surrogate import AdaptiveSurrogate

    argv = [*PROBLEMS[problem], "--dimension", str(d), "--leaf-degree", str(degree), "--error-tolerance", str(tol),
            "--max-depth", "0", "--store-samples"]  # fmt: skip
    if problem == "plane_wave":
        argv += plane_normal(d)
    if problem == "rastrigin" and d == 5:
        argv += ["--domain-low", "-1.5", "--domain-high", "1.5"]
    args = parse_args(argv)
    validate_args(args)
    low, high = BENCHMARK_DEFAULT_INTERVALS[args.benchmark]
    low = low if args.domain_low is None else args.domain_low
    high = high if args.domain_high is None else args.domain_high
    domain = BoxDomain.hypercube(d, low, high)
    target, _ = make_benchmark_target(args, domain)
    run = build_tree_benchmark(args, domain, target)
    root = run.tree.root
    X, y = np.asarray(root.metadata["X"]), np.asarray(root.metadata["y"]).reshape(-1)

    def f(x):
        return xp.asarray(np.asarray(target(np.asarray(to_host(x))), dtype=np.float64).reshape(-1, 1))

    config = TrainingConfig(
        algorithm="art", domain=(domain.bounds[:, 0], domain.bounds[:, 1]), rtol=tol, error_metric="pointwise",
        max_levels=0, seed=args.seed, local_model={1: "affine_lsq", 2: "quadratic_lsq"}[degree],
        art=ARTSettings(sample_multiplier=args.sample_multiplier),
    )  # fmt: skip
    model = AdaptiveSurrogate()
    model.train(xp.asarray(X), config, Evaluator(f_fn=f))

    def library_predict(Z):
        return np.asarray(to_host(model.predict_route_local(xp.asarray(Z))[0]))[:, 0]

    rng = np.random.default_rng(args.test_seed)
    Xt = rng.uniform(low, high, size=(10_000, d))
    p_ref, p_af = run.tree.predict(Xt), library_predict(Xt)
    if degree == 1:
        A, At = np.hstack([np.ones((len(X), 1)), X]), np.hstack([np.ones((len(Xt), 1)), Xt])
        p_np = At @ np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        p_np = None

    def train_err(pred):
        return float(np.max(np.abs(pred - y) / np.maximum(np.abs(y), args.relative_error_floor)))

    e_ref, e_af = train_err(run.tree.predict(X)), train_err(library_predict(X))
    scale = float(np.max(np.abs(p_ref)))
    reasons = model.leaf_status_counts()["terminal_reason"]
    accepted_af = reasons.get("accepted", 0) == 1
    accepted_ref = str(root.status) == "tolerance_met"
    evals = int(model.profiling_stats["total_f_evals"])
    near_tol = abs(e_ref - tol) <= 1e-6 * tol  # a decision this close may flip on the ridge
    out = {
        "problem": problem, "dim": d, "degree": degree, "tol": tol, "n_root": len(X),
        "pred_diff_rel": float(np.max(np.abs(p_ref - p_af))) / scale,
        "numpy_diff_rel": None if p_np is None else float(np.max(np.abs(p_ref - p_np))) / scale,
        "train_err_reference": e_ref, "train_err_library": e_af, "accepted_reference": accepted_ref, "accepted_library": accepted_af,
        "ref_status": str(root.status), "library_reasons": reasons, "library_evals": evals,
    }  # fmt: skip
    ok = (out["pred_diff_rel"] <= 1e-6 and math.isclose(e_ref, e_af, rel_tol=1e-6, abs_tol=1e-8)
          and (accepted_ref == accepted_af or near_tol) and evals <= len(X) + 1)  # fmt: skip
    if tol < 1e-3:
        ok = ok and accepted_ref and accepted_af  # the exactly representable case must be accepted
    out["result"] = "pass" if ok else "FAIL"
    return out


def run_slab(width: float, degree: int) -> dict:
    """Both codes' leaf fit of f = 2 + x_0 + x_1 / width on [0, 1] x [0, width] (exactly affine)."""
    from examples.test_helpers import make_model_template

    from src import to_host, xp
    from src.config import ARTSettings, TrainingConfig
    from src.core.evaluation import Evaluator
    from src.surrogate import AdaptiveSurrogate

    lo, hi = np.zeros(2), np.array([1.0, width])
    p = 3 if degree == 1 else 6
    X = np.random.default_rng(0).uniform(lo, hi, (50 * p, 2))
    Xt = np.random.default_rng(1).uniform(lo, hi, (5000, 2))

    def g(Z):
        return 2.0 + Z[:, 0] + Z[:, 1] / width

    ref = make_model_template(degree, 0.0, True, "auto").fit(X, g(X))
    config = TrainingConfig(
        algorithm="art", domain=(tuple(lo), tuple(hi)), rtol=1e-6, error_metric="pointwise", max_levels=0,
        local_model={1: "affine_lsq", 2: "quadratic_lsq"}[degree], art=ARTSettings(),
    )  # fmt: skip
    model = AdaptiveSurrogate()
    model.train(xp.asarray(X), config, Evaluator(f_fn=lambda x: xp.asarray(g(np.asarray(to_host(x))).reshape(-1, 1))))
    p_af = np.asarray(to_host(model.predict_route_local(xp.asarray(Xt))[0]))[:, 0]
    e_ref = float(np.max(np.abs(ref.predict(Xt) - g(Xt)) / np.abs(g(Xt))))
    e_af = float(np.max(np.abs(p_af - g(Xt)) / np.abs(g(Xt))))
    out = {"check": "slab", "aspect": 1.0 / width, "degree": degree, "max_rel_err_reference": e_ref, "max_rel_err_library": e_af,
           "result_reference": "pass" if e_ref <= 1e-6 else "FAIL", "result_library": "pass" if e_af <= 1e-6 else "FAIL"}  # fmt: skip
    out["result"] = "pass" if out["result_reference"] == out["result_library"] == "pass" else "FAIL"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--art-path", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    sys.path.insert(0, str(a.art_path.resolve()))
    warnings.simplefilter("ignore")
    results = []
    for case in cases(a.smoke):
        try:
            r = run_case(*case, a.art_path)
        except Exception as exc:  # noqa: BLE001 - one broken case must not hide the others
            r = {"problem": case[0], "dim": case[1], "degree": case[2], "tol": case[3], "result": "FAIL", "error": repr(exc)}
        results.append(r)
        print("EXACT " + json.dumps(r, default=str), flush=True)
    for degree in (1, 2):
        for width in ((1.0, 1e-3) if a.smoke else (1.0, 1e-1, 1e-2, 1e-3, 1e-4)):
            r = run_slab(width, degree)
            results.append(r)
            print("EXACT " + json.dumps(r), flush=True)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(results, indent=1, default=str))
    failed = [r for r in results if r["result"] != "pass"]
    print(f"exact checks: {len(results) - len(failed)} pass, {len(failed)} fail")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
