# NOTE: THIS IS IMPORTANT
# Else the values will not agree with the original code.
from jax import config
config.update("jax_enable_x64", True)

import itertools
import jax
import jax.numpy as jnp

import original.slimplectic_GGL
from ggl import ggl, dereduce
from sympy import Symbol, Float

t_sym = Symbol('t')
ddt = Float(1)
q1 = Symbol('q')
qdot_1 = Symbol('qdot_1')


def compute_qdot_from_q(qi_vec, r, dt):
    dij = dereduce(ggl(r), dt)[2]
    return jax.numpy.matmul(dij, qi_vec)


def test_co():
    for (i, r) in itertools.product(range(20), range(10)):
        sympy_value, total_value = compute_values(i, r)

        print(r, total_value, sympy_value, total_value - sympy_value)

        assert jnp.allclose(
            total_value,
            sympy_value
        )


def compute_values(i, r):
    random_vec = jax.random.normal(
        key=jax.random.PRNGKey(i),
        shape=(r + 2,)
    )
    total_value = jnp.dot(
        dereduce(ggl(r), dt=1)[1],
        compute_qdot_from_q(random_vec, r=r, dt=1)
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


def test_r3():
    xs, ws, dij = ggl(3)
    xs_dr, ws_dr, dij_dr = dereduce((xs, ws, dij), dt=1)
    sympy_value, total_value = compute_values(0, 3)
    print("\n", total_value, sympy_value, total_value - sympy_value)
    assert jnp.allclose(
        total_value,
        sympy_value
    )


if __name__ == '__main__':
    test_r3()
