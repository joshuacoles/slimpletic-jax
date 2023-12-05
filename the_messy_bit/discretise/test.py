from jax import config, Array

from typing import Callable

from discretise import discretise_integral
import itertools
import jax
import jax.numpy as jnp

import original.slimplectic_GGL
from ggl import ggl, dereduce
from sympy import Symbol, Float, Expr

# NOTE: THIS IS IMPORTANT
# Else the values will not agree with the original code.
config.update("jax_enable_x64", True)

t_sym = Symbol('t')
ddt = Float(1)
q1 = Symbol('q')
qdot_1 = Symbol('qdot_1')


def compute_qidot_values(i, r):
    random_vec = jax.random.normal(
        key=jax.random.PRNGKey(i),
        shape=(r + 2,)
    )

    _, ws, dij = dereduce(ggl(r), dt=1)
    total_value = jnp.dot(
        ws,
        jax.numpy.matmul(dij, random_vec)
    )

    expr, table = original.slimplectic_GGL.GGL_Gen_Ld(
        tsymbol=t_sym,
        q_list=[q1],
        qprime_list=[qdot_1],
        ddt=ddt,
        r=r,
        L=qdot_1
    )

    sympy_value = float(expr.subs([
        (table[0][i], random_vec[i])
        for i in range(len(table[0]))
    ]))

    return sympy_value, total_value


def test_qidot_derivative_calculations():
    for (i, r) in itertools.product(range(20), range(10)):
        sympy_value, total_value = compute_qidot_values(i, r)

        print(r, total_value, sympy_value, total_value - sympy_value)

        assert jnp.allclose(
            total_value,
            sympy_value
        )


def setup_sympy_discrete_integral(
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
    sympy_result = setup_sympy_discrete_integral(
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
