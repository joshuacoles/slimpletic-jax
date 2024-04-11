from jax import Array, jit

import jax.lax
import jax.numpy as jnp
import scipy.special as sps
import numpy as np

__all__ = [
    'ggl',
    'dereduce',
]


def ggl(r: int) -> tuple[Array, Array, Array]:
    """
    Compute the GGL quadrature information for a given order.
    :param r: The order of the quadrature, must be an int >= 0.
    :return: A tuple of (xs, ws, derivative_matrix), where xs is a 1D array of the quadrature points, ws is a 1D array
    of the quadrature weights, and derivative_matrix is a 2D array representing the derivative matrix, which transforms
    from a vector of q values at the quadrature points to a vector of q dot values at the points, i.e.:

        (q_dot)_i = derivative_matrix @ (q)_i

    Note that the results returned are reduced by certain factors to match the original code these are:

        - ws = (dt / 2) * ws_reduced
        - derivative_matrix = (2 / dt) * derivative_matrix_reduced
    """
    legendre = jnp.array(sps.legendre(r + 1).c)

    # (x^2 - 1) * d/dx(P_{r + 1}(x))
    # QUESTION: Why is this multiplied by (x^2 - 1)?
    poly = np.polymul(
        np.array([1, 0, -1]),
        np.polyder(legendre)
    )

    # The roots of the polynomial are the quadrature points. These are not returned in order, so we sort them.
    # We also clip them to the range [-1, 1] to avoid numerical issues as we know statically that the roots are in this
    # range.
    xs = jnp.array(np.clip(np.sort(np.real(np.roots(poly))), -1, 1))

    legendre_at_xs = jnp.polyval(legendre, xs)
    ws = 2 / ((r + 1) * (r + 2) * legendre_at_xs ** 2)

    dij_a = -1 / 4 * (r + 1) * (r + 2)
    dij_b = -dij_a

    @jit
    def derivative_matrix_element(i_, j_):
        # For some reason we get floats here, convert to ints for indexing.
        i = i_.astype('int')
        j = j_.astype('int')

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
            lambda: legendre_at_xs[i] / (legendre_at_xs[j] * (xs[i] - xs[j]))
        )

    derivative_matrix = jnp.fromfunction(
        derivative_matrix_element,
        shape=(r + 2, r + 2),
    )

    return xs, ws, derivative_matrix


def dereduce(values, dt):
    """
    The values which are returned from ggl are reduced by certain factors to match the original code. Here we reverse
    this reduction.
    :param values: The tuple of values returned from ggl.
    :param dt: The timestep of the integrator.
    :return: The tuple of values, but with the reduction reversed.
    """
    xs, ws, derivative_matrix = values
    return xs, (dt / 2) * ws, (2 / dt) * derivative_matrix


def test_agreement_with_original_code():
    """
    Test that the GGL quadrature points, weights, and derivative matrix agree with the original code up to a certain
    tolerance.
    :return:
    """
    from .helpers import jax_enable_x64, floatify1, floatify2

    jax_enable_x64()
    for r in range(0, 20):
        from original.slimplectic_GGL import GGLdefs
        original_xs, original_ws, original_dij = GGLdefs(r)
        new_xs, new_ws, new_dij = ggl(r)

        assert jnp.allclose(floatify1(original_xs), new_xs)
        assert jnp.allclose(floatify1(original_ws), new_ws)
        assert jnp.allclose(floatify2(original_dij), new_dij)


def binary(num):
    import struct
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!d', num))


def test_range_of_xs():
    """
    This is a test to ensure that the quadrature points are in the range [-1, 1].
    """
    from .helpers import jax_enable_x64
    jax_enable_x64()

    for r in range(0, 20):
        new_xs, _, _ = ggl(r)

        assert jnp.all(new_xs >= -1.0)
        assert jnp.all(new_xs <= 1.0)


class GGLBundle:
    """
    A bundle of the values of the GGL quadrature method, which are used to compute the values of the quadrature points.

    The computation of these is kept separate from the DiscretisedSystem class as they are not expected to change, and
    are less amenable to JIT compilation and gradient computation.
    """

    def __init__(self, r: int):
        self.r = r
        self.xs, self.ws, self.dij = ggl(r)


jax.tree_util.register_pytree_node(
    GGLBundle,
    lambda bundle: ((bundle.r, bundle.xs, bundle.ws, bundle.dij), None),
    lambda data, _: GGLBundle(*data)
)
