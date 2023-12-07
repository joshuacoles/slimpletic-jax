from functools import reduce

import sympy
from jax import Array
from typing import Callable

import itertools
import jax
import jax.numpy as jnp

import original.slimplectic_GGL
from .ggl import ggl, dereduce
from sympy import Symbol, Float, Expr, S

from .discretise_integral import discretise_integral

from .helpers import jax_enable_x64

jax_enable_x64()

t_sym = Symbol('t')
ddt = Float(1)
q1 = Symbol('q')
qdot_1 = Symbol('qdot_1')


def compute_qidot_values(qi_vec, r):
    _, ws, dij = dereduce(ggl(r), dt=1)

    jax_value = jnp.dot(
        ws,
        jax.numpy.matmul(dij, qi_vec)
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
        (table[0][i], qi_vec[i])
        for i in range(len(table[0]))
    ]))

    return sympy_value, jax_value


def test_qidot_derivative_calculations():
    """
    Tests that the qi_dot values calculated by sympy and jax are the same for different values of r.

    This test uses 20 random vectors to ensure that the test is not passing by chance so may not be repeatable.
    """
    for (i, r) in itertools.product(range(20), range(10)):
        random_vec = jax.random.normal(
            key=jax.random.PRNGKey(i),
            shape=(r + 2,)
        )

        sympy_value, jax_value = compute_qidot_values(random_vec, r)

        assert jnp.allclose(
            jax_value,
            sympy_value
        )


def sympy_discretise_integral(
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

    def fn(q_vec: Array, _t: float = 0):
        # TODO: Test time dependence.
        assert q_vec.shape == (dof, r + 2)

        subs_list = []
        for i in range(len(table)):
            for j in range(len(table[0])):
                subs_list.append([table[i][j], q_vec.tolist()[i][j]])

        return float(expr.subs(subs_list))

    return fn


def test_discretise_integral():
    dt = 0.1

    for (random_key, dof, r) in itertools.product(range(5), range(5), range(5)):
        random_qi = jax.random.normal(
            key=jax.random.PRNGKey(0),
            shape=(dof, r + 2)
        )

        sympy_result = sympy_discretise_integral(
            r=r,
            dt=dt,
            dof=dof,

            # |q|^2 + |q_prime|^2
            expr_builder=lambda q_list, qprime_list, t_symbol: (
                    reduce(lambda sum, c: sum + c ** 2, q_list, S.Zero) +
                    reduce(lambda sum, c: sum + c ** 2, qprime_list, S.Zero)
            )
        )(random_qi)

        jax_result = discretise_integral(
            r=r,
            dt=dt,
            fn=lambda q_vec, q_dot_vec, t: jnp.dot(q_vec, q_vec) + jnp.dot(q_dot_vec, q_dot_vec)
        )[0](
            random_qi.transpose(),
            t0=0
        )

        assert jnp.allclose(sympy_result, jax_result)
