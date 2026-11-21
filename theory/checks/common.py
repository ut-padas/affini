"""Helpers shared by the theory memo's numerical checks: JSON output, Chebyshev coefficients and decay fits."""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def write(name: str, data: dict) -> Path:
    """Write one check's results to data/<name>.json (sorted keys, so reruns diff cleanly)."""
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=float) + "\n")
    return path


def cheb_points(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """The n + 1 Chebyshev-Lobatto points cos(pi j / n) mapped to [a, b] (j = 0 first, so x runs from b to a)."""
    t = np.cos(np.pi * np.arange(n + 1) / n)
    return 0.5 * (a + b) + 0.5 * (b - a) * t


def cheb_coefficients(values: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients c_0..c_n of the interpolant through values at cheb_points(n) (along axis 0)."""
    v = np.asarray(values, dtype=float)
    n = v.shape[0] - 1
    ext = np.concatenate([v, v[-2:0:-1]], axis=0)  # even extension: the DCT-I as a real FFT
    c = np.real(np.fft.fft(ext, axis=0))[: n + 1] / n
    c[0] /= 2
    c[n] /= 2
    return c


def decay_rate(c: np.ndarray, floor: float = 1e-12, k_min: int = 2) -> dict:
    """
    rho_eff from |c_k| ~ C rho^-k: a least-squares fit of log of the coefficients' running upper envelope (the
    largest |c_j| for j >= k, so parity zeros do not bias it) over k_min <= k <= the last k above floor * max |c|.
    """
    a = np.abs(np.asarray(c, dtype=float))
    env = np.maximum.accumulate(a[::-1])[::-1]
    top = env.max()
    if top == 0:
        return {"rho": float("inf"), "k_used": 0}
    keep = np.nonzero(env > floor * top)[0]
    k_hi = int(keep.max()) if len(keep) else 0
    ks = np.arange(k_min, k_hi + 1)
    if len(ks) < 3:
        return {"rho": float("inf"), "k_used": int(len(ks))}
    slope = np.polyfit(ks, np.log(env[ks]), 1)[0]
    return {"rho": float(np.exp(-slope)), "k_used": int(len(ks))}


def rho_from_halfwidth(b: float) -> float:
    """The Bernstein ellipse parameter rho of [-1, 1] whose semi-minor axis is b: (rho - 1/rho) / 2 = b."""
    return float(b + np.sqrt(b * b + 1.0))
