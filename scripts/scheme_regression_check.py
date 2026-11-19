"""
Differential regression check for the schemes: train each scheme on one small, fast case and save, or
compare against a saved reference, its oracle evaluations and its predictions (and refusals) on a fixed
set of points. A refactor must reproduce every case bit for bit; a deliberate numerical change is listed
and justified separately.

    SURROGATE_DEVICE=cpu SURROGATE_MPI=off PYTHONPATH=. .venv/bin/python scripts/scheme_regression_check.py \\
        --save /tmp/schemes-new --compare /tmp/schemes-ref

Every case runs in a few seconds on one core; the whole check in well under a minute. BLAS runs on one
thread: the hub expansion's Gram products round differently with the thread count.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # before NumPy loads BLAS

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from src.problems import FUNCTIONS  # noqa: E402
from src.schemes import (  # noqa: E402
    chebtree,
    chebtt,
    chebtt_adaptive,
    hub,
    kernel,
    sparsegrid,
    tabulation,
)

N_QUERY = 400


def problem(name: str, d: int):
    """(f on host arrays, lo, hi) of a registered problem: (N,) values, or (N, m) scaled components."""
    p = FUNCTIONS[name]
    lo, hi = p.domain(d)
    return p.values, lo, hi


def finite_difference_jacobian(f, lo, hi):
    """A Jacobian by central differences at 1e-6 of the width, standing in for an exact one."""
    h = 1e-6 * (hi - lo)

    def jacobian(X):
        cols = []
        for i in range(X.shape[1]):
            e = np.zeros(X.shape[1])
            e[i] = h[i]
            cols.append((f(X + e).reshape(len(X), -1) - f(X - e).reshape(len(X), -1)) / (2 * h[i]))
        return np.stack(cols, axis=1)

    return jacobian


def linear_step_map():
    """A stiff linear 3-state step map (eigenvalues -1, -10, -100), its exact step, and 300 visited states."""
    rng = np.random.default_rng(7)
    Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    P = expm(Q @ np.diag([-1.0, -10.0, -100.0]) @ Q.T * 0.05)
    Z = [rng.uniform(-1, 1, (10, 3))]
    for _ in range(29):
        Z.append(Z[-1] @ P.T)
    return (lambda X: X @ P.T), np.concatenate(Z)


def case_chebtt():
    f, lo, hi = problem("genz_gaussian", 3)
    return chebtt.build(f, lo, hi, 0.01, seed=1), lo, hi


def case_chebtt_vector():
    f, lo, hi = problem("rosenbrock_gradient", 3)
    return chebtt.build_vector(f, lo, hi, 0.02, 3, seed=1), lo, hi


def case_chebtt_metric():
    f, lo, hi = problem("wave_pair", 3)
    return chebtt.build_vector(f, lo, hi, 0.05, 2, rotate=True, seed=1), lo, hi


def case_chebtt_adaptive():
    f, lo, hi = problem("oblique_jump", 2)
    return chebtt_adaptive.build(f, lo, hi, 0.05, seed=1), lo, hi


def case_chebtree():
    f, lo, hi = problem("genz_continuous", 2)
    return chebtree.build(f, lo, hi, 0.01), lo, hi


def case_chebtree_rotated():
    f, lo, hi = problem("genz_oscillatory", 3)
    return chebtree.build_rotated(f, lo, hi, 0.01, seed=1), lo, hi


def case_sparsegrid():
    f, lo, hi = problem("genz_product_peak", 3)
    return sparsegrid.build(f, lo, hi, 0.01, seed=1), lo, hi


def case_hub():
    f, lo, hi = problem("shared_latent", 4)
    return hub.build(f, lo, hi, 0.05, n0=500, max_evals=4000), lo, hi


def case_hub_jacobian():
    f, lo, hi = problem("shared_latent", 4)
    return hub.build(f, lo, hi, 0.05, jac=finite_difference_jacobian(f, lo, hi), n0=500, max_evals=4000), lo, hi


def case_kernel():
    f, lo, hi = problem("gaussian", 2)
    return kernel.build(f, lo, hi, 0.05, seed=1, max_evals=3000), lo, hi


def case_tabulation():
    step, states = linear_step_map()
    model = tabulation.build(step, states, 1e-3, seed=1)
    return model, states.min(axis=0), states.max(axis=0)


CASES = {name[5:]: fn for name, fn in globals().items() if name.startswith("case_")}


def run(name: str) -> dict:
    """One case's arrays: evaluations, predictions and (where the model refuses) refusals on fixed points."""
    t0 = time.perf_counter()
    model, lo, hi = CASES[name]()
    Q = np.random.default_rng(12345).uniform(lo, hi, (N_QUERY, len(lo)))
    out = {"evals": np.array(model.evals), "predict": np.asarray(model.predict(Q), dtype=np.float64)}
    if hasattr(model, "refuses"):
        out["refuses"] = np.asarray(model.refuses(Q))
    out["seconds"] = np.array(time.perf_counter() - t0)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", type=Path, help="directory to write this run's arrays to")
    parser.add_argument("--compare", type=Path, help="directory of a reference run to compare against")
    parser.add_argument("--cases", default=",".join(CASES), help="comma-separated subset of " + ",".join(CASES))
    a = parser.parse_args(argv)
    failed = []
    for name in a.cases.split(","):
        arrays = run(name)
        line = f"{name:18s} {int(arrays['evals']):8,d} evals  {float(arrays['seconds']):5.1f} s"
        if a.save:
            a.save.mkdir(parents=True, exist_ok=True)
            np.savez(a.save / f"{name}.npz", **arrays)
        if a.compare:
            ref = np.load(a.compare / f"{name}.npz")
            diffs = [k for k in ref.files if k != "seconds" and not (k in arrays and np.array_equal(ref[k], arrays[k]))]
            line += "  identical" if not diffs else f"  DIFFERS in {', '.join(diffs)}"
            if diffs:
                failed.append(name)
        print(line, flush=True)
    if failed:
        print(f"{len(failed)} case(s) differ: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
