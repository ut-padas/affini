"""
Differential regression check for the training/inference pipeline.

Trains small, fixed problems through the public ``AFINNISurrogate`` API for every
``SurrogateMode`` x error metric x output dimension (plus a few extra paths: oracle,
precomputed Y, uniform-grid roots, second-order Taylor, tiny batches), predicts on
fixed queries, and stores every leaf array, prediction, trust mask and counter.

It only uses arguments that also exist on older commits, so the same file can be run
on two checkouts and compared:

    # from the old checkout (or a git worktree of it)
    AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. python /path/to/regression_check.py --save old
    # from the new checkout
    AFINNI_DEVICE=cpu AFINNI_MPI=off PYTHONPATH=. python scripts/regression_check.py --save new --compare old

``--compare`` exits non-zero if any case that ran on the reference fails now, if any
counter (leaves, evaluations, generated points) or trust mask differs, or if any array
differs by more than ``--rtol`` (default 0: bit-identical, the right gate for a CPU
refactor). On a GPU, rounding order can differ between commits; use e.g. ``--rtol 1e-12``
there: decisions (counters and masks) must still match exactly. Cases that failed on the
reference but run now are reported as fixed, not as failures.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))

from src import IS_CUPY_BACKEND, _to_host, xp  # noqa: E402
from src.config import ClusteringConfig# noqa: E402
from src.core.evaluation import Evaluator# noqa: E402
from src.components.modes import SurrogateMode# noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402

warnings.simplefilter("ignore", DeprecationWarning)

H_SCALAR = np.array([[2.0, 0.2], [0.2, 6.0]])
LEAF_ARRAYS = ("X_src", "f_src", "Qs_src", "J_src", "Hs_src", "A_fits_src", "b_fits_src")


def problem(fd):
    """A quadratic (fd=1) or two quadratics with different signs/scales (fd=2)."""

    def f(x):
        a = x[:, 0] ** 2 + 3 * x[:, 1] ** 2 + 0.2 * x[:, 0] * x[:, 1] + 1
        return a[:, None] if fd == 1 else xp.stack((a, -2 * a + x[:, 0]), axis=1)

    def g(x):
        a = xp.stack((2 * x[:, 0] + 0.2 * x[:, 1], 6 * x[:, 1] + 0.2 * x[:, 0]), axis=1)
        if fd == 1:
            return a[:, None, :]
        return xp.stack((a, -2 * a + xp.asarray([1.0, 0.0])), axis=1)

    def h(x):
        return xp.tile(xp.asarray(H_SCALAR), (len(x), 1, 1))

    return f, g, h


def cases():
    """(name, fd, ClusteringConfig kwargs) for every case."""
    base = dict(
        atol=0.15, rtol=0.15, num_initial_clusters=4, min_pts_per_cluster=16,
        max_clusters_est=128, max_levels=12, batch_size=200,
        use_work_stealing=False, save_all_gen_training_data=False,
    )  # fmt: skip
    for fd in (1, 2):
        for mode in SurrogateMode:
            for metric in ("L2", "Linf"):
                yield f"{fd}_{mode.value}_{metric}", fd, dict(base, mode=mode, error_metric=metric)
    mode = SurrogateMode.AFFINE_COV_HESS
    yield "extra_oracle_taylor", 1, dict(base, mode=SurrogateMode.TAYLOR_COV_HESS, _oracle=True)
    yield "extra_precomputed_Y", 2, dict(base, mode=mode, _Y=True)
    yield "extra_uniform_grid", 1, dict(base, mode=mode, initial_clustering_method="uniform_grid")
    yield "extra_tiny_batches", 1, dict(base, mode=mode, batch_size=7)
    yield "extra_second_order", 1, dict(base, mode=SurrogateMode.TAYLOR_LSQ_HESS, store_hess=True)
    yield "extra_tight_tol", 1, dict(base, mode=mode, atol=0.02, rtol=0.02, max_clusters_est=4)
    # One tolerance only, and one value per output component (newer API: these fail
    # on older checkouts, which compare() reports as fixed rather than as failures).
    yield "extra_atol_only", 2, dict(base, mode=mode, atol=0.15, rtol=None)
    yield "extra_rtol_only", 2, dict(base, mode=mode, atol=None, rtol=0.15)
    yield "extra_vector_tol", 2, dict(base, mode=mode, atol=(0.15, 0.3), rtol=(0.1, 0.2), error_metric="L2")


def run(out_prefix):
    """Train and predict every case; save arrays to ``<prefix>.npz`` and counters/errors to ``<prefix>.json``."""
    rng = np.random.default_rng(82)
    X = xp.asarray(rng.uniform(-1, 1, (512, 2)))
    Q = xp.asarray(rng.uniform(-1.2, 1.2, (257, 2)))
    arrays, report = {}, {}
    for name, fd, kwargs in cases():
        f, g, h = problem(fd)
        oracle, use_Y = kwargs.pop("_oracle", False), kwargs.pop("_Y", False)
        fns = dict(oracle_fn=lambda x: (f(x), g(x))) if oracle else dict(f_fn=f, g_fn=g)
        t0 = time.perf_counter()
        try:
            s = AFINNISurrogate()
            s.train(X, ClusteringConfig(**kwargs), Evaluator(hessian_fn=h, **fns), Y_train=f(X) if use_Y else None)
            train_time = time.perf_counter() - t0
            for attr in LEAF_ARRAYS:
                v = getattr(s, attr, None)
                if v is not None:
                    v = _to_host(v)
                    arrays[f"{name}|{attr}"] = v.reshape(len(v), -1)
            for k, coarse, strict in ((1, 10, False), (4, 10, False), (4, 10, True), (2000, 2000, False)):
                pred, mask = s.predict(Q, k_nn=k, coarse_factor=coarse, strict_trust_region=strict, batch_size=37)
                arrays[f"{name}|pred_{k}_{strict}"] = _to_host(pred)
                arrays[f"{name}|mask_{k}_{strict}"] = _to_host(mask)
            ps = s.profiling_stats
            report[name] = dict(
                leaves=int(s.N_src), evals=int(ps["total_f_evals"]),
                generated=int(ps["num_generated_pts"]), train_s=round(train_time, 4),
            )  # fmt: skip
        except Exception as e:  # recorded, so a baseline failure is visible
            report[name] = dict(error=f"{type(e).__name__}: {e}")
    np.savez(out_prefix + ".npz", **arrays)
    meta = dict(backend="cupy" if IS_CUPY_BACKEND else "numpy", cases=report)
    Path(out_prefix + ".json").write_text(json.dumps(meta, indent=2))
    n_err = sum("error" in r for r in report.values())
    print(f"ran {len(report)} cases ({n_err} errors) on {meta['backend']} -> {out_prefix}.npz/.json")
    return meta


def compare(new_prefix, ref_prefix, rtol):
    """Print every difference from the reference run; True if none breaks the rules in the module docstring."""
    new, ref = (json.loads(Path(p + ".json").read_text())["cases"] for p in (new_prefix, ref_prefix))
    na, ra = np.load(new_prefix + ".npz"), np.load(ref_prefix + ".npz")
    failures, fixed, worst = [], [], 0.0
    for name, r in ref.items():
        n = new.get(name, {"error": "case missing"})
        if "error" in r:
            if "error" not in n:
                fixed.append(name)
            continue
        if "error" in n:
            failures.append(f"{name}: now fails: {n['error']}")
            continue
        for key in ("leaves", "evals", "generated"):
            if n[key] != r[key]:
                failures.append(f"{name}: {key} {r[key]} -> {n[key]}")
        for k in (k for k in ra.files if k.startswith(name + "|")):
            if k not in na.files:
                failures.append(f"{k}: missing")
                continue
            a, b = ra[k], na[k]
            if a.shape != b.shape:
                failures.append(f"{k}: shape {a.shape} -> {b.shape}")
            elif a.dtype == bool:
                if (a != b).any():
                    failures.append(f"{k}: {int((a != b).sum())} mask entries differ")
            elif not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.all(np.isfinite(b)):
                # NaN compares as "no difference" in the relative error below.
                failures.append(f"{k}: {int((~np.isfinite(b)).sum())} non-finite values (reference: {int((~np.isfinite(a)).sum())})")
            elif a.size:
                err = float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))
                worst = max(worst, err)
                if err > rtol:
                    failures.append(f"{k}: max relative difference {err:.3e} > {rtol:g}")
    print(f"compared {len(ref)} cases; max relative array difference {worst:.3e}")
    if fixed:
        print(f"fixed (failed on reference, run now): {len(fixed)}: {', '.join(fixed)}")
    for msg in failures:
        print("FAIL", msg)
    print("RESULT:", "FAIL" if failures else "PASS")
    return not failures


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--save", required=True, help="output prefix for this run (.npz/.json)")
    p.add_argument("--compare", help="reference prefix written by an earlier --save")
    p.add_argument("--rtol", type=float, default=0.0, help="allowed relative array difference")
    args = p.parse_args()
    run(args.save)
    if args.compare and not compare(args.save, args.compare, args.rtol):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
