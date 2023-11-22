import jax.numpy as jnp
from sympy import Symbol, lambdify

from discretise import compute_qdotvec_from_qvec
from ggl import ggl, dereduce
from original.slimplectic_GGL import DM_Sum, GGLdefs, GGL_q_Collocation_Table


def test_derivative_computation():
    r = 3
    dof = 4
    dt = 0.1

    qvec = jnp.arange((r + 2) * dof).reshape(((r + 2), dof))
    derivative_matrix = dereduce(ggl(r), 0.1)[2]

    qdotvec = compute_qdotvec_from_qvec(qvec, derivative_matrix)

    DM = GGLdefs(r)[2]
    q_list = [Symbol(f'q_{i}') for i in range(r + 2)]
    q_table = GGL_q_Collocation_Table(q_list, r + 2)
    ddt_symbol = Symbol('ddt')

    dphidt_Table = []
    for qs in q_table:
        dphidt_Table.append([DM_Sum(DMvec, qs) * 2 / ddt_symbol for DMvec in DM])

    qdotvec_original = lambdify([*q_list, ddt_symbol], dphidt_Table)(*qvec.transpose(), dt)

    assert jnp.allclose(qdotvec, qdotvec_original)
