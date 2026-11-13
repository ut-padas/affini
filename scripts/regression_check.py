"""
Differential regression check for the training/inference pipeline.

Trains small, fixed problems through the public ``AFINNISurrogate`` API for every
``SurrogateMode`` x error metric x output dimension (plus a few extra paths: oracle,
precomputed Y, uniform-grid roots, second-order Taylor, tiny batches) and ART
(every split search, root type, sampler, quadratic leaves, fallback leaves), predicts on
fixed queries, and stores every array a saved model holds, every prediction, trust mask
and counter.

Under ``mpiexec`` every rank trains and rank 0 writes and compares, so the same check
covers the scatter of the initial clusters and the merge of the leaves (cases do not
use work stealing, whose timing would change the tree between runs).

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

from src import IS_CUPY_BACKEND, get_mpi_rank, to_host, xp  # noqa: E402
from src.components.modes import SurrogateMode  # noqa: E402
from src.config import ARTSettings, TrainingConfig  # noqa: E402
from src.core.evaluation import Evaluator  # noqa: E402
from src.surrogate import AFINNISurrogate  # noqa: E402

warnings.simplefilter("ignore", DeprecationWarning)

H_SCALAR = np.array([[2.0, 0.2], [0.2, 6.0]])
ART_DOMAIN = ([-1.0, -1.0], [1.0, 1.0])


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


def wavy(fd):
    """A function ART cannot fit with one leaf (the quadratic is exact for quadratic leaves)."""

    def f(x):
        a = xp.sin(3 * x[:, 0]) * xp.cos(2 * x[:, 1]) + x[:, 0] ** 2
        return a[:, None] if fd == 1 else xp.stack((a, xp.cos(2 * x[:, 0] + x[:, 1])), axis=1)

    return f


def model_arrays(s: AFINNISurrogate) -> dict[str, np.ndarray]:
    """Every numeric array ``save_model`` would write (leaves, local-model parameters, routing, leaf status)."""
    out = {k: getattr(s, k) for k in ("X_src", "f_src", "Qs_src", "E_cluster_abs_src", "E_cluster_rel_src", "counts_src")}
    out.update({f"param:{k}": v for k, v in s.leaf_params.items()})
    if s.routing is not None:
        out.update({f"route:{k}": v for k, v in s.routing.arrays().items()})
    out.update({f"leaf:{k}": v for k, v in s.art_leaves.items()})
    return {k: to_host(v) for k, v in out.items() if v is not None}


def cases():
    """(name, fd, TrainingConfig kwargs) for every case."""
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
    art_base = dict(
        atol=1e-2, rtol=1e-2, algorithm="art", domain=ART_DOMAIN, num_initial_clusters=4,
        max_levels=10, use_work_stealing=False, save_all_gen_training_data=False,
    )  # fmt: skip
    rs = dict(sample_multiplier=10, validation_samples=10)
    yield "art_soft_box", 1, dict(art_base, art=ARTSettings(**rs))
    yield "art_bins_grid", 2, dict(art_base, art=ARTSettings(**rs, split_search="bins", roots="grid"))
    yield "art_gap_voronoi", 1, dict(art_base, art=ARTSettings(**rs, split_search="gap", roots="voronoi"))
    yield "art_convex_sampling", 1, dict(art_base, art=ARTSettings(sample_multiplier=10, sampling="convex_combination"))
    yield "art_quadratic", 1, dict(art_base, local_model="quadratic_lsq", art=ARTSettings(**rs))
    yield "art_tiny_batches", 1, dict(art_base, batch_size=97, art=ARTSettings(**rs, soft_iters=5))
    yield "art_fallback_leaves", 2, dict(art_base, max_levels=3, art=ARTSettings(**rs, soft_iters=5))


def run(out_prefix):
    """Train and predict every case; save arrays to ``<prefix>.npz`` and counters/errors to ``<prefix>.json``."""
    rng = np.random.default_rng(82)
    X = xp.asarray(rng.uniform(-1, 1, (512, 2)))
    Q = xp.asarray(rng.uniform(-1.2, 1.2, (257, 2)))
    arrays, report = {}, {}
    for name, fd, kwargs in cases():
        art_mode = kwargs.get("algorithm") == "art"
        f, g, h = problem(fd)
        if art_mode:
            f = wavy(fd)
        oracle, use_Y = kwargs.pop("_oracle", False), kwargs.pop("_Y", False)
        fns = dict(oracle_fn=lambda x, f=f, g=g: (f(x), g(x))) if oracle else dict(f_fn=f, g_fn=g)
        t0 = time.perf_counter()
        try:
            s = AFINNISurrogate()
            s.train(X, TrainingConfig(**kwargs), Evaluator(hessian_fn=h, **fns), Y_train=f(X) if use_Y else None)
            train_time = time.perf_counter() - t0
            for attr, v in model_arrays(s).items():
                arrays[f"{name}|{attr}"] = v.reshape(len(v), -1) if v.ndim else v
            if art_mode:
                pred, mask = s.predict(Q)  # routed down the tree
                arrays[f"{name}|pred_route"], arrays[f"{name}|mask_route"] = to_host(pred), to_host(mask)
                counters = {**s.profiling_stats.get("art", {}), **s.profiling_stats.get("terminal_reasons", {})}
                arrays[f"{name}|art_counters"] = np.asarray([counters[k] for k in sorted(counters)], dtype=np.int64)
            else:
                for k, coarse, strict in ((1, 10, False), (4, 10, False), (4, 10, True), (2000, 2000, False)):
                    pred, mask = s.predict(Q, k_nn=k, coarse_factor=coarse, strict_trust_region=strict, batch_size=37)
                    arrays[f"{name}|pred_{k}_{strict}"] = to_host(pred)
                    arrays[f"{name}|mask_{k}_{strict}"] = to_host(mask)
            ps = s.profiling_stats
            report[name] = dict(
                leaves=int(s.N_src), evals=int(ps["total_f_evals"]),
                generated=int(ps["num_generated_pts"]), train_s=round(train_time, 4),
            )  # fmt: skip
        except Exception as e:  # recorded, so a baseline failure is visible
            report[name] = dict(error=f"{type(e).__name__}: {e}")
    meta = dict(backend="cupy" if IS_CUPY_BACKEND else "numpy", cases=report)
    if get_mpi_rank() != 0:  # every rank holds the same model; rank 0 writes
        return meta
    np.savez(out_prefix + ".npz", **arrays)
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
            elif a.dtype == bool or a.dtype.kind in "iu":  # masks, counters, keys and statuses: exact
                if (a != b).any():
                    failures.append(f"{k}: {int((a != b).sum())} entries differ")
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
    if get_mpi_rank() == 0 and args.compare and not compare(args.save, args.compare, args.rtol):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
