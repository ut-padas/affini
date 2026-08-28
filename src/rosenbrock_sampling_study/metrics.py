"""Evaluation and resource metrics for study checkpoints."""

from __future__ import annotations

import resource
import time
from typing import Any

import numpy as np

from src import _to_device, _to_host
from src.surrogate import AFINNISurrogate


def _safe_quantile(values: np.ndarray, probability: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, probability))


def evaluate(
    surrogate: AFINNISurrogate,
    points: np.ndarray,
    truth: np.ndarray,
    tolerance: float,
    membership_k: int,
    prediction_batch_size: int = 5_000,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate prediction, coverage, calibration, and pointwise error."""

    start = time.perf_counter()
    predictions, unguaranteed = surrogate.predict(
        query_points=_to_device(points),
        k_nn=1,
        coarse_factor=membership_k,
        strict_trust_region=False,
        batch_size=prediction_batch_size,
    )
    elapsed = time.perf_counter() - start
    pred = np.asarray(_to_host(predictions), dtype=np.float64)
    out_mask = np.asarray(_to_host(unguaranteed), dtype=bool)
    truth = np.asarray(truth, dtype=np.float64)

    diff = pred - truth
    point_error = np.max(np.abs(diff) / (np.abs(truth) + 1e-12), axis=1)
    in_mask = ~out_mask
    in_error = point_error[in_mask]
    out_error = point_error[out_mask]
    n_test = len(points)
    n_out = int(out_mask.sum())
    n_in = n_test - n_out
    n_in_bad = int((in_error > tolerance).sum())

    def relative_l2(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(
            np.linalg.norm(diff[mask]) / (np.linalg.norm(truth[mask]) + 1e-12)
        )

    metrics = {
        "n_test": n_test,
        "n_out": n_out,
        "n_in": n_in,
        "p_out": n_out / n_test,
        "n_in_bad": n_in_bad,
        "p_in_bad_unconditional": n_in_bad / n_test,
        "p_bad_given_in": n_in_bad / n_in if n_in else float("nan"),
        "l2_relative_all": relative_l2(np.ones(n_test, dtype=bool)),
        "l2_relative_in": relative_l2(in_mask),
        "l2_relative_out": relative_l2(out_mask),
        "point_error_q50_all": _safe_quantile(point_error, 0.50),
        "point_error_q90_all": _safe_quantile(point_error, 0.90),
        "point_error_q95_all": _safe_quantile(point_error, 0.95),
        "point_error_q99_all": _safe_quantile(point_error, 0.99),
        "point_error_max_all": _safe_quantile(point_error, 1.00),
        "point_error_q50_in": _safe_quantile(in_error, 0.50),
        "point_error_q90_in": _safe_quantile(in_error, 0.90),
        "point_error_q95_in": _safe_quantile(in_error, 0.95),
        "point_error_q99_in": _safe_quantile(in_error, 0.99),
        "point_error_max_in": _safe_quantile(in_error, 1.00),
        "point_error_q50_out": _safe_quantile(out_error, 0.50),
        "point_error_q90_out": _safe_quantile(out_error, 0.90),
        "point_error_q95_out": _safe_quantile(out_error, 0.95),
        "point_error_q99_out": _safe_quantile(out_error, 0.99),
        "point_error_max_out": _safe_quantile(out_error, 1.00),
        "evaluation_seconds": elapsed,
    }
    return metrics, out_mask


def coverage_transitions(
    previous_out: np.ndarray | None, current_out: np.ndarray
) -> dict[str, int | float]:
    if previous_out is None:
        return {
            "n_in_to_in": 0,
            "n_in_to_out": 0,
            "n_out_to_in": 0,
            "n_out_to_out": 0,
            "p_in_to_out": float("nan"),
            "p_out_to_in": float("nan"),
        }
    in_to_in = int(((~previous_out) & (~current_out)).sum())
    in_to_out = int(((~previous_out) & current_out).sum())
    out_to_in = int((previous_out & (~current_out)).sum())
    out_to_out = int((previous_out & current_out).sum())
    n_previous_in = int((~previous_out).sum())
    n_previous_out = int(previous_out.sum())
    return {
        "n_in_to_in": in_to_in,
        "n_in_to_out": in_to_out,
        "n_out_to_in": out_to_in,
        "n_out_to_out": out_to_out,
        "p_in_to_out": in_to_out / n_previous_in if n_previous_in else float("nan"),
        "p_out_to_in": out_to_in / n_previous_out if n_previous_out else float("nan"),
    }


def peak_host_memory_gib() -> float:
    """Return Linux maximum resident set size in GiB."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def gpu_memory_gib() -> float:
    try:
        import cupy as cp

        return cp.get_default_memory_pool().used_bytes() / (1024.0**3)
    except Exception:
        return float("nan")
