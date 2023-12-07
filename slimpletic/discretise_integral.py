import jax
import jax.numpy as jnp
from jax import Array

from .ggl import ggl, dereduce

__all__ = [
    'discretise_integral',
]


def test_range_of_quadrature_points():
    """
    This is a test to ensure that the quadrature points are in the range [-1, 1].
    """
    from random import random

    for r in range(1, 5):
        dt = random()
        _, t_quadrature_offsets = discretise_integral(
            r=r,
            dt=dt,
            fn=lambda _, __, ___: 0
        )

        assert jnp.all(t_quadrature_offsets >= 0)
        assert jnp.all(t_quadrature_offsets <= dt)


# A tad redundant, but given things are being weird, I want to be sure.
def test_coincidence_of_quadrature_points():
    from original.slimplectic_GGL import Symbol, lambdify
    from random import random

    t_symbol = Symbol('t')
    ddt_symbol = Symbol('dt')

    for r in range(0, 20):
        for _ in range(20):
            dt_val = random()
            xs, _, _ = ggl(r)  # No need to care about reduction here, xs is unreduced.
            t_list = [t_symbol + 0.5 * (1 + xi) * ddt_symbol for xi in xs]

            qua_values = lambdify([t_symbol, ddt_symbol], t_list)(0, dt_val)
            qua_neu_values = (1 + xs) * dt_val / 2

            assert jnp.allclose(jnp.array(qua_values), qua_neu_values)


def test_coincidence_of_discretised_fn():
    from original.slimplectic_GGL import GGL_Gen_Ld
    from sympy import Symbol

    GGL_Gen_Ld(
        tsymbol=Symbol('t'),
        ddt=Symbol('dt'),
    )


def discretise_integral(
        r: int,
        dt: float,
        fn: callable,
) -> tuple[callable, Array]:
    """
    Discretise the integral corresponding with Eq. 7 in the paper. This performs two operations:

    1. It evaluates the integral at the points specified by the Gauss-Legendre quadrature rule, weighted by the
       corresponding weights.
    2. It applies the derivative matrix to the q vector for each dof, to obtain the value of the derivatives at each
       quadrature point.

    :param dt: The time step of the integrator.
    :param r: The order of the method.
    :param fn: The function to discretise.
    :return: A callable which takes a vector of q values and returns a vector of q dot values. As well as a vector of
             offsets to add to the time to obtain the quadrature points.
    """
    xs, ws, dij = dereduce(ggl(r), dt)

    # Eq. 4
    # These are the offsets **within** [t0, t0 + dt] which we evaluate the function at to compute the integral.
    t_quadrature_offsets = (1 + xs) * dt / 2

    def discretised_fn(qi_vec, t0):
        # Eq. 6
        qidot_vec = jax.numpy.matmul(dij, qi_vec)

        t_quadrature_values = t0 + t_quadrature_offsets

        # Eq. 7
        return jnp.dot(ws, jax.vmap(
            fn
        )(
            qi_vec,
            qidot_vec,
            t_quadrature_values,
        ))

    return discretised_fn, t_quadrature_offsets
