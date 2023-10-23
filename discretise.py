from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit, Array
from jax.experimental import checkify

from dj import compute_quadrature_scheme

m = 1.0
k = 1.0


@jit
def conservative_lagrangian(q, q_dot, t):
    return 0.5 * m * jnp.dot(q_dot, q_dot) - 0.5 * k * jnp.dot(q, q)


@jit
def non_conservative_lagrangian(q_p, q_m, v_p, v_m, t):
    return -q_p * v_m


def weighted_sum(
        ws: Array,
        fs: Array,
):
    return jnp.dot(ws, fs)


jax.debug.print("weighted_sum {}", weighted_sum(
    jnp.array([1.0, 2.0]),
    jnp.array([1.0, 3.0]),
))


def q_dot_from_q_through_phi(
        qn_i: Array,
        derivative_matrix: Array,
):
    """
    Computes values of dq_i/dt (the derivative of q_i with respect to time), for the quadrature times, using the values
    of qn^{(i)} at the quadrature times and the derivative_matrix D_{ij}. This is given by equation (6) in the paper.

    :param qn_i: The vector of qn^{(i)} values, for a fixed n, indexed by the interior index (i).
    :param derivative_matrix: The derivative matrix, see equation (5) in the paper, and compute_quadrature_scheme.
    :return: The values of dq_i/dt at the quadrature times.
    """

    # TODO: In the original code there is an additional factor of 2 / ddt here, why?
    return jax.numpy.matmul(derivative_matrix, qn_i)


jax.debug.print("q_dot_from_q_through_phi {}", q_dot_from_q_through_phi(
    jnp.array([1.0, 2.0, 3.0, 4.0]),
    compute_quadrature_scheme(2, 0.1)[2],
))


@partial(jit, static_argnums=(0, 1))
def evaluate_discrete_non_conservative_lagrangian(
        lagrangian: Callable[[Array, Array, Array, Array, float], float],
        _r: int,

        _collation_points: Array,
        weights: Array,
        derivative_matrix: Array,

        qn_plus_i: Array,
        qn_minus_i: Array,
        tn_i: Array,
):
    qn_plus_dot_i = q_dot_from_q_through_phi(qn_plus_i, derivative_matrix)
    qn_minus_dot_i = q_dot_from_q_through_phi(qn_minus_i, derivative_matrix)

    method_a = jnp.dot(
        weights,
        jax.vmap(
            lambda stack: lagrangian(stack[0], stack[1], stack[2], stack[3], stack[4]),
            in_axes=0,
            out_axes=0,
        )(
            jnp.stack([qn_plus_dot_i, qn_minus_dot_i, qn_plus_dot_i, qn_minus_dot_i, tn_i], axis=1)
        )
    )

    return method_a  # , method_b


@partial(jit, static_argnums=(0, 1))
def evaluate_discrete_lagrangian(
        lagrangian: Callable[[Array, Array, float], float],
        _r: int,

        _collation_points: Array,
        weights: Array,
        derivative_matrix: Array,

        qn_i: Array,
        tn_i: Array,
):
    qn_dot_i = q_dot_from_q_through_phi(qn_i, derivative_matrix)

    method_a = jnp.dot(
        weights,
        jax.vmap(
            lambda stack: lagrangian(stack[0], stack[1], stack[2]),
            in_axes=0,
            out_axes=0,
        )(
            jnp.stack([qn_i, qn_dot_i, tn_i], axis=1)
        )
    )

    # # TODO: Is this better expressed as some from of vstack and vmap and sum, maybe more parallelisable?
    # method_b = jax.lax.fori_loop(
    #     0,
    #     r + 2,
    #     # TODO: In the original code there is an additional factor of 0.5 * ddt here (cancelling the 2 / ddt above in
    #     #  q_dot_from_q_through_phi) why?
    #     # TODO: Check these parameters are right, should they be offset by q_n?
    #     lambda i, acc: acc + weights[i] * lagrangian(qn_i[i], qn_dot_i[i], tn_i[i]),
    #     0.0
    # )

    return method_a  # , method_b


jax.debug.print("DL {}", evaluate_discrete_lagrangian(
    conservative_lagrangian,
    2,
    *compute_quadrature_scheme(2, 0.1),
    jnp.array([1.0, 2.0, 3.0, 4.0]),
    jnp.array([1.0, 2.0, 3.0, 4.0]),
))

jax.debug.print("NC_DL {}", evaluate_discrete_non_conservative_lagrangian(
    non_conservative_lagrangian,
    2,
    *compute_quadrature_scheme(2, 0.1),
    jnp.array([1.0, 2.0, 3.0, 4.0]),
    jnp.array([1.0, 2.0, 3.0, 4.0]),
    jnp.array([1.0, 2.0, 3.0, 4.0]),
))
