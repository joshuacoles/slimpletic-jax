import jax
import numpy as np
from jax import lax, numpy as jnp


def integrate_inner(
        i,
        state,
):
    q, pi, ts = state
    dt = ts[i] - ts[i - 1]

    q = q + dt * pi
    pi = pi - dt * q
    return q, pi, ts


def integrate(
        q0,
        pi0,
        ts,
):
    q_final, p_final, _ = lax.fori_loop(
        0,
        ts.shape[0],
        integrate_inner,
        (q0, pi0, ts)
    )

    return q_final, p_final


print(integrate(
    1,
    2,
    jnp.array([0.0, 1.0, 2.0, 3.0])
))
