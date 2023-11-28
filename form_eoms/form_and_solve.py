from typing import Callable

import jax
import jax.numpy as jnp
import jaxopt
from jax import Array

from discretise.fn_3 import discretise_integral


def make_residue(fn: Callable[[Array, float], float]) -> Callable[[Array, float], Array]:
    """
    Computes the residue of the equation of motion system for the provided discrete lagrangian.

    Given the equation of motion f(x) = g(x), the residue is defined as r(x) = f(x) - g(x), which we
    seek to make equal to zero.

    :param fn: The discrete lagrangian, in the form (q_vec, t) -> float.
    :return: The residue function, having signature (q_vec, t) -> [float; r + 1].
    """
    derivatives = jax.grad(fn, argnums=0)

    def residue(q_vec, t):
        dfdx = derivatives(q_vec, t)
        inner = dfdx[1:-1]
        return jnp.append(inner, dfdx[0] - dfdx[-1])

    return residue


def single_step(
        t0: float,
        f_d: Callable[[Array, float], float],
        r: int,
        qi: Array = None,
):
    optimiser = jaxopt.GaussNewton(residual_fun=make_residue(f_d), verbose=True)

    opt_res = optimiser.run(qi, t0)

    # Return the values of q_i representing the path of the system which minimises the residue.
    return opt_res.params


def fill_out_initial(initial, r):
    return jnp.repeat(initial[jnp.newaxis, :], r + 2, axis=0)


def test_fill_out_initial():
    assert jnp.array_equal(
        fill_out_initial(
            initial=jnp.array([1, 2, 3]),
            r=3
        ),
        jnp.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ])
    )

def iterate(
        q0: Array,
        t0: float,
        dt: float,
        t_sample_count: int,
        r: int,
        lagrangian: Callable[[Array, Array, float], float],
):
    lagrangian_d = discretise_integral(
        fn=lagrangian,
        r=r,
        dt=dt
    )

    t_samples = t0 + jnp.arange(t_sample_count) * dt

    def scan_body_2(
            previous_q,
            t_value
    ):
        jax.debug.print("previous_state {}", previous_q)
        jax.debug.print("t_value {}", t_value)

        qi_values = single_step(
            qi=fill_out_initial(previous_q, r=r),
            t0=t_value,
            r=r,
            f_d=lagrangian_d
        )

        jax.debug.print("qi_values {}", qi_values)

        return qi_values[-1], qi_values

    _, results = jax.lax.scan(
        f=scan_body_2,
        xs=t_samples,
        init=q0,
    )

    return results
