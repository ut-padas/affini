import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController
import time


# Definition of Lorenz system
def lorenz96_jax(t, x, F):
    x_minus_1 = jnp.roll(x, 1)
    x_minus_2 = jnp.roll(x, 2)
    x_plus_1 = jnp.roll(x, -1)
    dxdt = (x_plus_1 - x_minus_2) * x_minus_1 - x + F
    return dxdt


# Define flowmap
def lorenz96_flowmap_jax(y0, forcing, tf):
    rhs = ODETerm(lorenz96_jax)
    solver = Dopri5()
    controller = PIDController(rtol=1e-3, atol=1e-3)
    sol = diffeqsolve(
        rhs,
        solver,
        t0=0,
        t1=tf,
        dt0=0.01,
        y0=y0,
        args=forcing,
        stepsize_controller=controller,
    )
    return sol.ys


batch_solve = jax.vmap(lorenz96_flowmap_jax, in_axes=(0, None, None))
lorenz96_flowmap_jax_batched = jax.jit(batch_solve)


if __name__ == "__main__":
    # Create a batch of initial conditions
    key = jax.random.PRNGKey(42)
    N = 100000
    d = 5
    batch_y0 = jax.random.uniform(key, (N, d), minval=-5.0, maxval=5.0)

    warmup_y0 = jnp.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    lorenz96_flowmap_jax_batched(warmup_y0, forcing=1.0, tf=1.0)

    start_time = time.perf_counter()
    results = lorenz96_flowmap_jax_batched(batch_y0, 1.0, 1.0)
    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    print(f"Result shape: {results.shape}")

    # Test the exact same thing with scipy
    # sol_scipy = solve_ivp(lambda t, y: lorenz96_np(t, y, np.array(batch_p[test_row])), [0, 10], np.array(batch_y0[test_row]), method='RK45', rtol=1e-6)
    # print(f'Result with scipy: {sol_scipy.y[:,-1]}')
