from typing import Callable

import jax
import jax.numpy as jnp
import jaxopt
from jax import Array

from .discretise_integral import discretise_integral
from .helpers import fill_out_initial

__all__ = [
    'iterate'
]


def make_residue(fn: Callable[[Array, float], float]) -> Callable[[Array, float], Array]:
    """
    Computes the residue of the equation of motion system for the provided discrete lagrangian.

    Given the equation of motion f(x) = g(x), the residue is defined as r(x) = f(x) - g(x), which we
    seek to make equal to zero.

    :param fn: The discrete lagrangian, in the form (q_vec, t) -> float.
    :return: The residue function, having signature (q_vec, t) -> [float; r + 1].
    """
    derivatives = jax.grad(fn, argnums=0)

    def residue(q_vec, t, pi0):
        dfdx = derivatives(q_vec, t)

        # Eq 13(c), we set the derivative wrt to each interior point to zero
        eq13c_residues = dfdx[1:-1]

        # Eq 13(a), we set the derivative wrt to the final point to negative of pi0
        eq13a_residue = pi0 + dfdx[0]

        return jnp.append(
            eq13c_residues,
            eq13a_residue
        )

    return residue


def iterate(
        q0: Array,
        pi0: Array,
        t0: float,
        dt: float,
        t_sample_count: int,
        r: int,
        lagrangian: Callable[[Array, Array, float], float],
        debug: bool = False
):
    lagrangian_d, t_quadrature_offsets = discretise_integral(
        fn=lagrangian,
        r=r,
        dt=dt
    )

    # These are the values of t which we will sample the solution at.
    t_samples = t0 + jnp.arange(t_sample_count) * dt

    # TODO: TEST THIS
    def compute_pi_next(qi_values, t_value):
        # Eq 13(b)
        derivatives = jax.grad(lagrangian_d, argnums=0)
        v = derivatives(qi_values, t_value)[-1]
        return v

    # Set up the residual function and optimiser once as they can be reused.
    residue = make_residue(lagrangian_d)
    optimiser = jaxopt.GaussNewton(residual_fun=residue)

    def compute_next(
            previous_state,
            t_value
    ):
        (previous_q, previous_pi) = previous_state

        optimiser_result = optimiser.run(
            fill_out_initial(previous_q, r=r),
            t_value,
            previous_pi
        )

        qi_values = optimiser_result.params

        jax.debug.print("qi_values {}", qi_values)

        # q_{n, r + 1} = q_{n + 1, 0}
        q_next = qi_values[-1]
        pi_next = compute_pi_next(qi_values, t_value)
        jax.debug.print("pi_current {} pi_next {}", pi0, pi_next)
        next_state = (q_next, pi_next)

        return next_state, next_state

    _, (q, pi) = jax.lax.scan(
        f=compute_next,
        xs=t_samples,
        init=(q0, pi0),
    )

    # We need to add the initial values back into the results.
    q_with_initial = jnp.insert(q, 0, q0)
    pi_with_initial = jnp.insert(pi, 0, pi0)

    if debug:
        from slimpletic.ggl import dereduce, ggl

        debug_info = {
            't_samples': t_samples,
            't_quadrature_offsets': t_quadrature_offsets,
            'lagrangian_d': lagrangian_d,
            'residue': residue,
            'compute_pi_next': compute_pi_next,
            'ggl_data': dereduce(ggl(r), dt)
        }

        return q_with_initial, pi_with_initial, debug_info
    else:
        return q_with_initial, pi_with_initial
