"""Lorenz96 ODE solver using JAX (Diffrax) for massive parallelism on GPUs.

Features:
- JIT-compiled for XLA acceleration
- Implicit Adaptive Solver (Kvaerno5) for stiff tolerances
- Forward/Reverse mode auto-differentiation for exact gradients
- Pure functional design
"""

from functools import partial

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from src import IS_CUPY_BACKEND, Array

# We use jax.numpy internally, but input/output might be CuPy/NumPy (via Array),
# So we cast as needed. We generally expect inputs to be convertible to JAX arrays.

# Enable 64-bit globally for ODE precision if not already set by user config
jax.config.update("jax_enable_x64", True)


def _to_jax(x: Array) -> jax.Array:
    """Efficiently convert input array to JAX array (zero-copy if possible)."""
    # If already a JAX array, return as is
    if isinstance(x, jax.Array):
        return x

    # Use DLPack for zero-copy transfer from CuPy/Torch if available
    if hasattr(x, "__dlpack__"):
        return jax.dlpack.from_dlpack(x)

    # Fallback to standard conversion (e.g. from NumPy)
    return jnp.asarray(x)


def _from_jax(x: jax.Array) -> Array:
    """Efficiently convert JAX array to the active backend (CuPy/NumPy)."""
    # Block until computation is done to ensure data is ready before external access
    # (Crucial for correct timing and synchronization)
    x.block_until_ready()

    if IS_CUPY_BACKEND:
        import cupy as cp

        # Zero-copy transfer to CuPy via DLPack
        return cp.from_dlpack(x)

    # Copy to host NumPy array
    return np.asarray(x)


def lorenz96_rhs(t, y, args):
    """
    Lorenz 96 vector field in JAX.
    f_i = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F

    Args:
        t: Time (scalar) - unused (autonomous system)
        y: State vector (d,)
        args: Forcing constant F (scalar)
    """
    F = args
    y_p1 = jnp.roll(y, -1)
    y_m2 = jnp.roll(y, 2)
    y_m1 = jnp.roll(y, 1)
    return (y_p1 - y_m2) * y_m1 - y + F


# Define the global JIT-compiled kernel for the batch solver.
# This avoids recompiling the XLA graph for every call.
# in_axes=(0, None) means:
#   - X0 (arg 0): batch dimension is axis 0
#   - tf (arg 1): scalar (broadcasted)
@partial(jax.jit, static_argnames=("forcing", "dt0"))
def _batch_lorenz_solve_kernel(X0, tf, forcing=8.0, dt0=0.01):
    # 1. Setup ODE Term
    term = diffrax.ODETerm(lorenz96_rhs)

    # 2. Solver: Kvaerno5 (L-stable, implicit, order 5)
    solver = diffrax.Kvaerno5()

    # 3. Controller: Adaptive PID
    stepsize_controller = diffrax.PIDController(
        rtol=1e-6, atol=1e-6, pcoeff=0.3, icoeff=0.3, dcoeff=0.0
    )

    # Single trajectory solver (value only)
    def flow_map_single(x0, t_final):
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=t_final,
            dt0=dt0,
            y0=x0,
            args=forcing,
            stepsize_controller=stepsize_controller,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=100000,
            throw=False,  # Don't crash entire batch on one failure, though rare
        )
        return sol.ys[-1]  # type: ignore

    # Single trajectory solver (value + grad)
    # jacrev is preferred over jacfwd for implicit solvers involving custom_vjp loops
    def flow_and_grad_single(x0, t_final):
        val = flow_map_single(x0, t_final)
        jac = jax.jacrev(flow_map_single, argnums=0)(x0, t_final)
        return val, jac

    # Vectorize across batch
    return jax.vmap(flow_and_grad_single, in_axes=(0, None))(X0, tf)


def lorenz96_jax_oracle(
    X: Array, forcing: float = 8.0, tf: float = 0.1, batch_size: int = 100_000
) -> tuple[Array, Array]:
    """
    Computes both the flowmap F(x) and its gradient J(x) for Lorenz96 using JAX.

    This replaces `lorenz96_flowmap_oracle` with a GPU-accelerated massive batch version.
    Handles batching internally to avoid GPU OOM on massive datasets.

    Args:
        X: (N, d) input states
        forcing: Forcing constant F (default 8.0)
        tf: Final integration time (default 0.1)
        batch_size: Number of trajectories to process in parallel (chunk size).

    Returns:
        f_out: (N, d) final states
        g_out: (N, d, d) Jacobians d(x_tf)/d(x_0)
        (Both returned as Arrays on the active backend, e.g. CuPy)
    """
    # 1. Convert input to JAX Array (Zero-copy if CuPy)
    X_jax = _to_jax(X)
    N = X_jax.shape[0]

    # 2. Run Kernel in Batches
    if N <= batch_size:
        X_final_jax, J_final_jax = _batch_lorenz_solve_kernel(
            X_jax, tf, forcing=forcing
        )
    else:
        # Loop over chunks
        vals = []
        jacs = []
        # Process in chunks
        for i in tqdm(range(0, N, batch_size), desc="JAX Lorenz Oracle Batches"):
            # Slicing creates a view in JAX, efficient
            batch_X = X_jax[i : i + batch_size]
            # Calls JIT compiled kernel (compiles once for batch_size, maybe once more for remainder)
            v, j = _batch_lorenz_solve_kernel(batch_X, tf, forcing=forcing)
            vals.append(v)
            jacs.append(j)

        # Concatenate results
        X_final_jax = jnp.concatenate(vals, axis=0)
        J_final_jax = jnp.concatenate(jacs, axis=0)

    # 3. Convert back to active backend (Zero-copy if to CuPy)
    return _from_jax(X_final_jax), _from_jax(J_final_jax)


def lorenz96_jax_flowmap(X: Array, forcing: float = 8.0, tf: float = 0.1) -> Array:
    """Returns only the flowmap (final state) on active backend."""
    val_jax, _ = lorenz96_jax_oracle(X, forcing, tf)
    # Note: the oracle already converts to backend, so we are good.
    # Wait, oracle returns backend arrays now.
    return val_jax


def lorenz96_jax_grad(X: Array, forcing: float = 8.0, tf: float = 0.1) -> Array:
    """Returns only the Jacobian on active backend."""
    _, grad_jax = lorenz96_jax_oracle(X, forcing, tf)
    return grad_jax
