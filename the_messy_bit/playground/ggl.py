from jax import Array, jit

import jax.lax
import jax.numpy as jnp
import scipy.special as sps


def compute_reduced_quadrature_scheme(r: int) -> tuple[Array, Array, Array]:
    """
    Compute the collocation points, weights, and derivative matrix for the GGL quadrature. These are the reduced
    values which require the following transformation to get the full values:

    - `ws = (dt / 2) * ws_reduced`
    - `derivative_matrix = (2 / dt) * derivative_matrix_reduced`

    This reduction is done to align with the original code.
    :param r:
    :return:
    """
    c = sps.legendre(r + 3).c
    xs = jnp.real(jnp.roots(jnp.polyder(c)))
    leg_xs = jnp.polyval(c, xs)

    ws = 2 / ((r + 1) * (r + 2) * leg_xs ** 2)

    dij_a = -1 / 4 * (r + 1) * (r + 2)
    dij_b = -dij_a

    # TODO: Is there a nicer way to do this? Maybe jax.lax.switch?
    @jit
    def derivative_matrix_element(i_, j_):
        # For some reason we get floats here, convert to ints for indexing.
        i = i_.astype('int32')
        j = j_.astype('int32')

        return jax.lax.cond(
            jax.lax.eq(i, j),
            lambda: jax.lax.cond(
                jax.lax.eq(i, 0),
                lambda: dij_a,
                lambda: jax.lax.cond(
                    jax.lax.eq(i, r + 1),
                    lambda: dij_b,
                    lambda: 0.0,
                )
            ),
            lambda: leg_xs[i] / (leg_xs[j] * (xs[i] - xs[j]))
        )

    derivative_matrix = jnp.fromfunction(
        derivative_matrix_element,
        shape=(r + 2, r + 2),
    )

    return xs, ws, derivative_matrix


def compute_quadrature_scheme(r: int, dt: float) -> tuple[Array, Array, Array]:
    """
    Compute the collocation points, weights, and derivative matrix for the GGL quadrature.

    :param r: The number of interior points in the quadrature.
    :param dt: The timestep of the integrator.
    :return: The collocation points, weights, and derivative matrix as jax arrays.
    """
    c = sps.legendre(r + 3).c
    xs = jnp.real(jnp.roots(jnp.polyder(c)))
    leg_xs = jnp.polyval(c, xs)

    jax.debug.print("xs {}", xs)
    jax.debug.print("leg_xs {}", leg_xs)

    ws = dt / ((r + 1) * (r + 2) * leg_xs ** 2)

    jax.debug.print("ws {}", leg_xs)

    # TODO: Is there a nicer way to do this? Maybe jax.lax.switch?
    @jit
    def derivative_matrix_element(i_, j_):
        # For some reason we get floats here, convert to ints for indexing.
        i = i_.astype('int32')
        j = j_.astype('int32')

        return jax.lax.cond(
            jax.lax.eq(i, j),
            lambda: jax.lax.cond(
                jax.lax.eq(i, 0),
                lambda: -(r + 1) * (r + 2) / (2 * dt),
                lambda: jax.lax.cond(
                    jax.lax.eq(i, r + 1),
                    lambda: (r + 1) * (r + 2) / (2 * dt),
                    lambda: 0.0,
                )
            ),
            lambda: 2 * leg_xs[i] / (leg_xs[j] * (xs[i] - xs[j]) * dt)
        )

    derivative_matrix = jnp.fromfunction(
        derivative_matrix_element,
        shape=(r + 2, r + 2),
    )

    jax.debug.print("derivative_matrix {}", derivative_matrix)

    return xs, ws, derivative_matrix
