from jax import Array, jit

import jax.lax
import jax.numpy as jnp
import scipy.special as sps


def compute_quadrature_scheme(r: int, dt: float) -> tuple[Array, Array, Array]:
    """
    Compute the collocation points, weights, and derivative matrix for the GGL quadrature.

    :param r: The number of interior points in the quadrature over the phase space.
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


jax.numpy.set_printoptions(linewidth=200)
xs, ws, dij = compute_quadrature_scheme(5, 1)
