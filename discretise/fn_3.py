import jax
import jax.numpy as jnp

from ggl import ggl, dereduce


# Corresponds with Eq. 7 in the paper.
def discretise_integral(
        r: int,
        dt: float,
        fn: callable,
) -> callable:
    """
    :param r: The order of the method.
    :param fn: The function to discretise.
    :return: A callable which takes a vector of q values and returns a vector of q dot values.
    """

    xs, ws, dij = dereduce(ggl(r), dt)

    def discretised_fn(qi_vec, t0):
        qidot_vec = jax.numpy.matmul(dij, qi_vec)
        t_values = t0 + (1 + xs) * dt / 2

        return jnp.dot(ws, jax.vmap(
            fn
        )(
            qi_vec,
            qidot_vec,
            t_values,
        ))

    return discretised_fn


from jax import Array
from typing import Callable
from sympy import Expr, Symbol, Float
import original.slimplectic_GGL


def perform_sympy_calc(
        r: int,
        dt: float,
        dof: int,
        expr_builder: Callable[[list[Symbol], list[Symbol], Symbol], Expr],
):
    q_list = [Symbol(f'q{i}') for i in range(dof)]
    qprime_list = [Symbol(f'qdot{i}') for i in range(dof)]
    t_symbol = Symbol('t')

    expr, table = original.slimplectic_GGL.GGL_Gen_Ld(
        tsymbol=t_symbol,
        q_list=q_list,
        qprime_list=qprime_list,
        r=r,
        ddt=Float(dt),
        L=expr_builder(q_list, qprime_list, t_symbol)
    )

    def fn(q_vec: Array, t: float = 0):
        assert q_vec.shape == (dof, r + 2)

        subs_list = []
        for i in range(len(table)):
            for j in range(len(table[0])):
                subs_list.append([table[i][j], q_vec.tolist()[i][j]])

        return float(expr.subs(subs_list))

    return fn


def test_discretise_integral():
    sympy_result = perform_sympy_calc(
        r=3,
        dt=0.1,
        dof=3,
        expr_builder=lambda q_list, qprime_list, t_symbol: (
                q_list[0] ** 2 + q_list[1] ** 2 + q_list[2] ** 2 +
                qprime_list[0] ** 2 + qprime_list[1] ** 2 + qprime_list[2] ** 2
        )
    )(
        jnp.array([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ])
    )

    jax_result = discretise_integral(
        r=3,
        dt=0.1,
        fn=lambda q_vec, q_dot_vec, t: jnp.dot(q_vec, q_vec) + jnp.dot(q_dot_vec, q_dot_vec)
    )(
        jnp.array([
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]).transpose(),
        t0=0
    )

    assert jnp.allclose(sympy_result, jax_result)
