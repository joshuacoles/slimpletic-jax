import jax
import jax.numpy as jnp

from .ggl import ggl, dereduce

__all__ = [
    'discretise_integral',
]


def discretise_integral(
        r: int,
        dt: float,
        fn: callable,
) -> callable:
    """
    Discretise the integral corresponding with Eq. 7 in the paper. This performs two operations:

    1. It evaluates the integral at the points specified by the Gauss-Legendre quadrature rule, weighted by the
       corresponding weights.
    2. It applies the derivative matrix to the q vector for each dof, to obtain the value of the derivatives at each
       quadrature point.

    :param dt: The time step of the integrator.
    :param r: The order of the method.
    :param fn: The function to discretise.
    :return: A callable which takes a vector of q values and returns a vector of q dot values.
    """
    xs, ws, dij = dereduce(ggl(r), dt)

    def discretised_fn(qi_vec, t0):
        # Eq. 6
        qidot_vec = jax.numpy.matmul(dij, qi_vec)

        # Eq. 4
        t_values = t0 + (1 + xs) * dt / 2

        # Eq. 7
        return jnp.dot(ws, jax.vmap(
            fn
        )(
            qi_vec,
            qidot_vec,
            t_values,
        ))

    return discretised_fn
