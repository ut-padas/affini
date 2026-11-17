"""
Batched ensemble integration of autonomous ODEs: many independent trajectories x' = f(x) from t = 0 to
t1, each with its own adaptive step, by an explicit embedded Runge-Kutta pair. The same algorithm runs
as a CUDA kernel (``src/kernels/ode.cu``) and as Numba on the CPU, so a trajectory's result does not
depend on the batch it is in (a single solver over the stacked batch would share one step size).
"""

from dataclasses import dataclass

import numba as nb
import numpy as np

from src import IS_CUPY_BACKEND, Array, kernels, xp


@dataclass(frozen=True)
class ButcherTableau:
    """An embedded explicit Runge-Kutta pair: ``A`` (s, s) lower triangular, weights ``b`` of the
    propagated solution, ``e = b - b_hat`` for the error estimate, ``order`` sets the step-size exponent
    ``1 / order``, and ``fsal`` (first same as last) means the last stage is f at the new solution."""

    A: np.ndarray
    b: np.ndarray
    e: np.ndarray
    order: int
    fsal: bool


# Dormand & Prince (1980), the 5(4) pair of Hairer, Norsett & Wanner, Solving ODEs I, Table II.5.2.
_b5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_b4 = np.array(
    [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)
DOPRI5 = ButcherTableau(
    A=np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ]),
    b=_b5, e=_b5 - _b4, order=5, fsal=True,
)  # fmt: skip

# Components one GPU lane carries (ODE_MAX_COMPONENTS_PER_LANE in ode.cu): up to this d a trajectory
# gets one thread, above it one warp of 32 lanes, so up to 32 * this.
ODE_THREAD_MAX_D = 32
ODE_BLOCK = 128


@nb.njit(cache=True, inline="always")
def _lorenz96_rhs(x, j, d, params):  # pragma: no cover - compiled
    jp1 = 0 if j + 1 == d else j + 1
    jm1 = d - 1 if j == 0 else j - 1
    jm2 = j - 2 if j >= 2 else j - 2 + d
    return (x[jp1] - x[jm2]) * x[jm1] - x[j] + params[0]


@nb.njit(cache=True, inline="always")
def _scaled_sq(v, scale):  # pragma: no cover - compiled
    r = v / scale
    return r * r


@nb.njit(error_model="numpy", inline="always")
def _trajectory_cpu(
    rhs, y, t1, rtol, atol, A, b, e, order, fsal, params
):  # pragma: no cover - compiled
    """
    The CPU twin of ``ensemble_rk`` in ode.cu for one trajectory: advances ``y`` in place to ``t1``
    with the same initial-step rule, error norm and step-size control, components in order; returns
    False when the trajectory fails. Inlined into each cached entry point below, where ``rhs`` becomes
    that entry point's global right-hand side.
    """
    d = y.shape[0]
    s = A.shape[0]
    yn = np.empty(d)
    x = np.empty(d)
    k = np.empty((s, d))
    for j in range(d):
        k[0, j] = rhs(y, j, d, params)
    sy = sf = 0.0
    for j in range(d):
        sc = atol + rtol * abs(y[j])
        sy += _scaled_sq(y[j], sc)
        sf += _scaled_sq(k[0, j], sc)
    d0, d1 = np.sqrt(sy / d), np.sqrt(sf / d)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    for j in range(d):
        x[j] = y[j] + h0 * k[0, j]
    sd = 0.0
    for j in range(d):
        sd += _scaled_sq(rhs(x, j, d, params) - k[0, j], atol + rtol * abs(y[j]))
    d2 = np.sqrt(sd / d) / h0
    dm = max(d1, d2)
    h1 = max(1e-6, h0 * 1e-3) if dm <= 1e-15 else (0.01 / dm) ** (1.0 / order)
    h = min(min(100.0 * h0, h1), t1)
    t, steps, failed = 0.0, 0, False
    while t < t1:
        last = t + h >= t1
        if last:
            h = t1 - t
        for i in range(1, s):
            for j in range(d):
                acc = 0.0
                for q in range(i):
                    acc += A[i, q] * k[q, j]
                x[j] = y[j] + h * acc
            for j in range(d):
                k[i, j] = rhs(x, j, d, params)
        se = 0.0
        for j in range(d):
            accb = acce = 0.0
            for q in range(s):
                accb += b[q] * k[q, j]
                acce += e[q] * k[q, j]
            yn[j] = y[j] + h * accb
            se += _scaled_sq(h * acce, atol + rtol * max(abs(y[j]), abs(yn[j])))
        err = np.sqrt(se / d)
        steps += 1
        if err != err or steps > 1_000_000:
            failed = True
            break
        if err <= 1.0:
            t = t1 if last else t + h
            for j in range(d):
                y[j] = yn[j]
                if fsal:
                    k[0, j] = k[s - 1, j]
            if not fsal:
                for j in range(d):
                    k[0, j] = rhs(y, j, d, params)
            h *= min(10.0, max(0.2, 0.9 * max(err, 1e-300) ** (-1.0 / order)))
        else:
            h *= max(0.2, 0.9 * err ** (-1.0 / order))
    return not failed


# One cached entry point per right-hand side, parallel over trajectories, with the right-hand side
# named as a global. Numba caches neither closures nor functions that receive a dispatcher as a value
# (it becomes a dynamic global), so either form would recompile the integrator, about 2 s, in every
# process; inlining _trajectory_cpu keeps the call a plain global one.
@nb.njit(parallel=True, cache=True, error_model="numpy")
def _integrate_lorenz96_cpu(
    X0, X1, t1, rtol, atol, A, b, e, order, fsal, params
):  # pragma: no cover - compiled
    for traj in nb.prange(X0.shape[0]):
        y = X0[traj].copy()
        ok = _trajectory_cpu(
            _lorenz96_rhs, y, t1, rtol, atol, A, b, e, order, fsal, params
        )
        for j in range(y.shape[0]):
            X1[traj, j] = y[j] if ok else np.nan


# Right-hand sides by name: the CPU entry point and the CUDA entry points (one trajectory per
# thread, one per warp) in ode.cu.
_CPU = {"lorenz96": _integrate_lorenz96_cpu}
if IS_CUPY_BACKEND:
    _ode = kernels.module("ode")
    _CUDA = {
        "lorenz96": (
            _ode.get_function("ode_lorenz96_thread"),
            _ode.get_function("ode_lorenz96_warp"),
        )
    }


def integrate_ensemble(rhs: str, X0: Array, t1: float, params, rtol: float, atol: float,
                       tableau: ButcherTableau = DOPRI5) -> Array:  # fmt: skip
    """
    (N, d) states at ``t1`` of the N trajectories of x' = rhs(x) starting at the rows of ``X0`` at t = 0,
    in float64. Each trajectory has its own adaptive step (Hairer's initial step, the error norm over
    its d components, safety 0.9, step factors in [0.2, 10]) and its last step lands exactly on ``t1``,
    so its value does not depend on the other rows. A trajectory that fails (non-finite error, more
    than 10^6 steps) comes back as NaN. On the GPU d is at most 32 x 32.
    """
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    X0 = xp.ascontiguousarray(xp.asarray(X0, dtype=xp.float64))
    n, d = int(X0.shape[0]), int(X0.shape[1])
    X1 = xp.empty_like(X0)
    if n == 0:
        return X1
    if not IS_CUPY_BACKEND:
        _CPU[rhs](X0, X1, float(t1), float(rtol), float(atol), tableau.A, tableau.b, tableau.e,
                      tableau.order, tableau.fsal, params)  # fmt: skip
        return X1
    if d > 32 * ODE_THREAD_MAX_D:
        raise ValueError(
            f"integrate_ensemble on the GPU supports d <= {32 * ODE_THREAD_MAX_D}; got {d}"
        )
    lanes = 1 if d <= ODE_THREAD_MAX_D else 32
    per_block = ODE_BLOCK // lanes
    kernel = _CUDA[rhs][0 if lanes == 1 else 1]
    A, b, e, p = (
        xp.asarray(v, dtype=xp.float64)
        for v in (tableau.A, tableau.b, tableau.e, params)
    )
    kernel(((n + per_block - 1) // per_block,), (ODE_BLOCK,),
           (X0, X1, np.int64(n), np.int32(d), np.float64(t1), np.float64(rtol), np.float64(atol), A, b, e,
            np.int32(A.shape[0]), np.int32(tableau.fsal), np.int32(tableau.order), p),
           shared_mem=per_block * d * 8)  # fmt: skip
    return X1
