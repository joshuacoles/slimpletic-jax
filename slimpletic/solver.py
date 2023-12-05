from typing import Callable

import jax
import jax.numpy as jnp
import jaxopt
from jax import Array

from .discretise_integral import discretise_integral

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

        eq13a_residue = pi0 - dfdx[0]

        return jnp.append(
            eq13c_residues,
            eq13a_residue
        )

    return residue


def fill_out_initial(initial, r):
    return jnp.repeat(initial[jnp.newaxis, :], r + 2, axis=0)


def test_fill_out_initial():
    assert jnp.array_equal(
        fill_out_initial(
            initial=jnp.array([1, 2, 3]),
            r=3
        ),
        jnp.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
    )


def iterate(
        q0: Array,
        pi0: Array,
        t0: float,
        dt: float,
        t_sample_count: int,
        r: int,
        lagrangian: Callable[[Array, Array, float], float],
):
    lagrangian_d = discretise_integral(
        fn=lagrangian,
        r=r,
        dt=dt
    )

    # Eq 4
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

        jax.debug.print("previous_q {}\nprevious_pi {}", previous_q, previous_pi)
        jax.debug.print("t_value {}", t_value)

        optimiser_result = optimiser.run(
            fill_out_initial(previous_q, r=r),
            t0,
            pi0
        )

        qi_values = optimiser_result.params

        jax.debug.print("qi_values {}", qi_values)

        # q_{n, r + 1} = q_{n + 1, 0}
        q_next = qi_values[-1]
        pi_next = compute_pi_next(qi_values, t_value)
        next_state = (q_next, pi_next)

        return next_state, next_state

    _, results = jax.lax.scan(
        f=compute_next,
        xs=t_samples,
        init=(q0, pi0),
    )

    return results
