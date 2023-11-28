# NOTE: THIS IS IMPORTANT
# Else the values will not agree with the original code.
from jax import config
import jax
import jax.numpy as jnp

config.update("jax_enable_x64", True)

from form_eoms.form_and_solve import iterate

m = 10
k = 7


def lagrangian(q, dq, t):
    jax.debug.print("q {}; dq {}; t {}", q, dq, t)
    return 0.5 * 7 * jnp.dot(q, q) - 0.5 * jnp.dot(dq, dq)


a = iterate(
    q0=jnp.array([10.3]),
    pi0=jnp.array([1.0]),
    r=1,
    lagrangian=lagrangian,
    t_sample_count=100,
    dt=2,
    t0=0
)
