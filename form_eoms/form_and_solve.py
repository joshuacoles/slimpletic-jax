from typing import Callable

import jax
import jax.numpy as jnp
import jaxopt
from jax import Array


def make_residue(fn: Callable[[Array, float], float]) -> Callable[[Array, float], Array]:
    """
    Computes the residue of the equation of motion system for the provided discrete lagrangian.

    Given the equation of motion f(x) = g(x), the residue is defined as r(x) = f(x) - g(x), which we
    seek to make equal to zero.

    :param fn: The discrete lagrangian, in the form (q_vec, t) -> float.
    :return: The residue function, having signature (q_vec, t) -> [float; r + 1].
    """
    derivatives = jax.grad(fn, argnums=0)

    def residue(q_vec, t):
        dfdx = derivatives(q_vec, t)
        inner = dfdx[1:-1]
        return jnp.append(inner, dfdx[0] - dfdx[-1])

    return residue


def single_step(
        q0: Array,
        t0: float,
        f_d: Callable[[Array, float], float],
        r: int,
):
    # Repeated the provided initial condition r + 2 times.
    qi = jnp.repeat(q0[jnp.newaxis, :], r + 2, axis=0)

    optimiser = jaxopt.GaussNewton(residual_fun=make_residue(f_d), verbose=True)

    opt_res = optimiser.run(qi, t0)

    # Return the values of q_i representing the path of the system which minimises the residue.
    return opt_res.params
