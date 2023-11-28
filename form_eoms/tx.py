from jax import config
from jax import numpy as jnp

# NOTE: THIS IS IMPORTANT
# Else the values will not agree with the original code.
config.update("jax_enable_x64", True)

from form_eoms.form_and_solve import iterate

m = 10
k = 7


def lagrangian(q, dq, t):
    # jax.debug.print("q {}; dq {}; t {}", q, dq, t)
    return 0.5 * k * jnp.dot(q, q) - 0.5 * m * jnp.dot(dq, dq)


a = iterate(
    q0=jnp.array([12.0]),
    pi0=jnp.array([2.0]),
    r=1,
    lagrangian=lagrangian,
    t_sample_count=7000,
    dt=0.1,
    t0=0
)
