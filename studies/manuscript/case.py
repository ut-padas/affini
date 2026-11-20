"""
One case of the manuscript study through the library's public API: sample, train, predict with the
candidate search and with the exact search, measure, and write one result JSON.

    python -m studies.manuscript.case --case '<Case as JSON>' --out <result.json>

Run it under mpiexec for a case with ranks > 1 (SURROGATE_MPI=on); rank 0 samples, measures and
writes. A case that fails still writes its result, with the failure's class (the host memory guard,
the device running out of memory, any other error) and the stage it reached; one that the system
kills cannot, and the runner records it instead. Everything recorded is a by-product of the run or
a bookkeeping read, so the result costs no measurable time.
"""

import argparse
import contextlib
import json
import resource
import sys
import time
import zlib
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1


def functions(name: str):
    """(f, gradient or None) of a study function, from src.problems."""
    if name == "quadratic":
        from src.problems.anisotropic_quadratic_2 import (
            aniso_quadratic,
            aniso_quadratic_grad,
        )

        return aniso_quadratic, aniso_quadratic_grad
    if name == "gaussian":
        from src.problems.anisotropic_gaussian import (
            aniso_gaussian,
            aniso_gaussian_grad,
        )

        return aniso_gaussian, aniso_gaussian_grad
    if name == "rosenbrock":
        from src.problems.rosenbrock import rosenbrock, rosenbrock_grad

        return partial(rosenbrock, a=100.0), partial(rosenbrock_grad, a=100.0)
    if name == "wave":
        from src.problems.wave import wave, wave_grad

        return wave, wave_grad
    if name == "lorenz96":
        from src.problems.lorenz96_flowmap import lorenz96_flowmap

        return (lambda X: lorenz96_flowmap(X, 1.0, 0.1)[:, :1]), None
    raise ValueError(f"unknown function {name!r}")


def test_points(name: str, d: int, domain: tuple[float, float], n: int) -> np.ndarray:
    """The fixed test set of (function, d): its own seed, so every run of the pair is scored on the same points."""
    from studies.manuscript.spec import TEST_SEED

    rng = np.random.default_rng([TEST_SEED, zlib.crc32(name.encode()), d])
    return rng.uniform(domain[0], domain[1], size=(n, d))


def failure_class(err: BaseException) -> str:
    """The class the tables report a failed case under."""
    # cupy.cuda.memory.OutOfMemoryError (named, so CuPy is not imported) subclasses MemoryError: test it first
    if type(err).__name__ == "OutOfMemoryError":
        return "device_out_of_memory"
    if isinstance(err, MemoryError):
        return "host_memory_guard"
    return "error"


def device_peak_gib(peak: float) -> float:
    """The larger of ``peak`` and the bytes CuPy's pool holds now (0 on NumPy)."""
    from src import IS_CUPY_BACKEND, xp

    return max(peak, xp.get_default_memory_pool().total_bytes() / 2**30) if IS_CUPY_BACKEND else 0.0


def scores(preds, gt, outside, atol, rtol) -> dict:
    """compute_error_metrics of one prediction, with p_o (%), as plain JSON."""
    from src import xp
    from src.surrogate import compute_error_metrics

    m = compute_error_metrics(preds, gt, outside, atol=atol, rtol=rtol)
    m["p_o"] = float(100.0 * xp.mean(outside))
    return json.loads(json.dumps(m, default=float))


def train_once(case, X, Y, f, g, atol, rtol):
    """A trained AdaptiveSurrogate and its training time (collective under MPI)."""
    from src.components.modes import SurrogateMode
    from src.config import TrainingConfig
    from src.core.evaluation import Evaluator
    from src.surrogate import AdaptiveSurrogate
    from studies.manuscript.spec import MODES

    config = TrainingConfig(
        mode=SurrogateMode(MODES[case["mode"]]), atol=atol, rtol=rtol, error_metric="Linf",
        initial_clustering_method="kmeans", num_initial_clusters=case["k0"], min_pts_per_cluster=case["m"],
        use_work_stealing=case["work_stealing"], max_levels=1000, q_reg=1e-6, sampling_method="convex_combination",
        seed=case["seed"],
    )  # fmt: skip
    surrogate = AdaptiveSurrogate()
    t = time.perf_counter()
    surrogate.train(X, config, Evaluator(f_fn=f, g_fn=g if case["mode"] == "exact_gradient" else None), Y_train=Y)
    return surrogate, time.perf_counter() - t


def predict_both(surrogate, X_test, d: int, exact_max_dim: int | None):
    """((preds, outside, seconds) by the candidates, the same completed by the exact search or None)."""
    from studies.manuscript.spec import CANDIDATES

    t = time.perf_counter()
    p, o = surrogate.predict(query_points=X_test, k_nn=1, strict_trust_region=False, coarse_factor=CANDIDATES)
    cand = (p, o, time.perf_counter() - t)
    if exact_max_dim is not None and d > exact_max_dim:
        return cand, None
    t = time.perf_counter()
    p, o = surrogate.predict(query_points=X_test, k_nn=1, strict_trust_region=False, coarse_factor=CANDIDATES,
                             exact_outside=True)  # fmt: skip
    return cand, (p, o, time.perf_counter() - t)


def outside_points(surrogate, lo: float, hi: float, d: int, need: int, share: float, rng, cap: int = 10**8):
    """
    ``need`` points uniform in [lo, hi]^d outside every trust region (exact search), drawn in batches
    sized by ``share``, the outside share observed so far, so a small share costs bigger batches
    rather than more attempts. Returns (points, candidates drawn); fewer points when ``cap``
    candidates were not enough.
    """
    from src import to_device, to_host
    from studies.manuscript.spec import CANDIDATES

    found, drawn = [], 0
    while sum(len(x) for x in found) < need and drawn < cap:
        missing = need - sum(len(x) for x in found)
        batch = int(min(max(1.5 * missing / max(share, 1e-6), missing), 5_000_000, cap - drawn))
        cand = rng.uniform(lo, hi, size=(batch, d))
        outside = to_host(surrogate.filter_unguaranteed_local(to_device(cand), k_coarse=CANDIDATES, exact_outside=True))
        found.append(cand[outside])
        drawn += batch
        share = max(sum(len(x) for x in found) / drawn, 1e-6)
    pts = np.concatenate(found)[:need] if found else np.empty((0, d))
    return pts, drawn


def run_case(case: dict, out: Path, exact_max_dim: int | None = None, n_test: int | None = None) -> dict:
    """Run ``case`` (Case.as_dict()) and write its result to ``out`` on rank 0; returns the result."""
    from src import get_mpi_rank, to_device, to_host, xp
    from src.infra.backend import MPI
    from src.infra.records import failure, now, provenance, write_json
    from studies.manuscript.spec import FUNCTIONS, N_TEST, ROUND_SHARE

    rank = get_mpi_rank()
    t0 = time.perf_counter()
    result = {"schema": SCHEMA, "case": case, "status": "running", "stage": "setup", "failure": None}
    peak_device = 0.0
    try:
        result["provenance"] = provenance() if rank == 0 else None
        f, g = functions(case["function"])
        lo, hi = FUNCTIONS[case["function"]].domain
        d = case["d"]
        X_test = Y_test = X = Y = None
        atol = rtol = None
        if rank == 0:
            result["stage"] = "sampling"
            X_test = to_device(test_points(case["function"], d, (lo, hi), n_test or N_TEST))
            Y_test = f(X_test)
            xp.random.seed(case["seed"])
            X = xp.random.uniform(lo, hi, size=(case["n0"], d))
            Y = f(X)
            if case["atol"] is not None:  # the absolute-tolerance experiment: unscaled atol only
                atol = case["atol"]
            elif case["function"] == "lorenz96":  # absolute only, scaled by max|f| over the sample
                atol = tuple(float(case["tol"] * v) for v in to_host(xp.max(xp.abs(Y), axis=0)))
            else:
                atol = rtol = case["tol"]
        atol, rtol = MPI.COMM_WORLD.bcast((atol, rtol), root=0)
        result["tolerances"] = {"atol": atol, "rtol": rtol}
        rounds = []
        rng = np.random.default_rng(case["seed"] + 1)
        for r in range(case["rounds"] + 1):
            result["stage"] = "training"
            surrogate, train_s = train_once(case, X, Y, f, g, atol, rtol)
            peak_device = device_peak_gib(peak_device)
            result["stage"] = "prediction"
            cand, exact = predict_both(surrogate, X_test, d, exact_max_dim)
            peak_device = device_peak_gib(peak_device)
            if rank != 0:
                continue
            ps = surrogate.profiling_stats
            step = {
                "N_train": int(X.shape[0]), "N": ps.get("total_f_evals"), "N_c": int(surrogate.N_src),
                "num_levels": ps.get("num_levels"), "generated_points": ps.get("num_generated_pts"),
                "converged": bool(surrogate.converged), "unresolved_points": int(surrogate.num_unresolved_pts),
                "train_s": train_s, "predict_s": cand[2], "exact_s": exact[2] if exact else None,
                "candidates": scores(cand[0], Y_test, cand[1], atol, rtol),
                "exact": scores(exact[0], Y_test, exact[1], atol, rtol) if exact else None,
            }  # fmt: skip
            if r == case["rounds"]:
                result.update(step)
                result["level_stats"] = json.loads(json.dumps(ps.get("level_stats", []), default=float))
                result["effective_config"] = json.loads(json.dumps(ps.get("effective_config", {}), default=str))
            if case["rounds"]:
                rounds.append({k: v for k, v in step.items() if k not in ("candidates", "exact")} | {
                    "p_o": (step["exact"] or step["candidates"])["p_o"],
                    "e_in": (step["exact"] or step["candidates"])["in"]["l2_rel_error"],
                    "e": (step["exact"] or step["candidates"])["all"]["l2_rel_error"]})  # fmt: skip
                if r < case["rounds"]:
                    result["stage"] = "resampling"
                    t = time.perf_counter()
                    share = max(rounds[-1]["p_o"] / 100.0, 1e-6)
                    new, drawn = outside_points(surrogate, lo, hi, d, int(ROUND_SHARE * case["n0"]), share, rng)
                    rounds[-1].update(resample_s=time.perf_counter() - t, candidates_drawn=drawn, added=len(new))
                    if len(new) == 0:
                        rounds[-1]["stopped"] = "no point outside every trust region was found"
                        result.update(step)
                        break
                    Xn = to_device(new)
                    X = xp.concatenate([X, Xn])
                    Y = xp.concatenate([Y, f(Xn)])
        if case["rounds"]:
            result["rounds"] = rounds
        result["status"], result["stage"] = "done", "done"
    except Exception as err:  # every failure ends this case alone, with its class written down
        result["status"] = "failed"
        result["failure"] = {"class": failure_class(err), **failure(err)}
        with contextlib.suppress(Exception):  # what the pool held when the stage failed, e.g. a device out of memory
            peak_device = device_peak_gib(peak_device)
    result["wall_s"] = time.perf_counter() - t0
    result["memory"] = {"host_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                        "device_peak_gib": peak_device}  # fmt: skip
    if rank == 0:
        result["recorded"] = now()
        write_json(out, result, indent=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, help="the Case as JSON (Case.as_dict())")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exact-max-dim", type=int, default=None, help="skip the exact search above this d")
    parser.add_argument("--n-test", type=int, default=None, help="test points (default: spec.N_TEST)")
    a = parser.parse_args()
    result = run_case(json.loads(a.case), a.out, a.exact_max_dim, a.n_test)
    sys.exit(0 if result["status"] == "done" else 3)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
