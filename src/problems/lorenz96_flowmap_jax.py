import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, PIDController


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
