import jax.numpy as jnp
from jax import Array
from sympy import Symbol, lambdify

from ggl import ggl, dereduce
from original.slimplectic_GGL import DM_Sum


def compute_qdotvec_from_qvec(
        qvec: Array,
        derivative_matrix: Array,
):
    return jnp.matmul(derivative_matrix, qvec)


class GGLDefs:
    pass


def test_derivative_computation():
    r = 3
    dof = 4
    dt = 0.1

    qvec = jnp.arange(r * dof).reshape((r, dof))
    derivative_matrix = dereduce(ggl(r), 0.1)[2]

    qdotvec = compute_qdotvec_from_qvec(qvec, derivative_matrix)

    DM = GGLDefs(r)[2]

    q_symbol = Symbol('q')
    ddt_symbol = Symbol('ddt')
    qdotvec_original_sym = [DM_Sum(DMvec, q_symbol) * 2 / ddt_symbol for DMvec in DM]
    qdotvec_original = lambdify([q_symbol, ddt_symbol], qdotvec_original_sym)(qvec, dt)

    assert jnp.allclose(qdotvec, qdotvec_original)

