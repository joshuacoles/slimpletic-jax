import jax
import jax.numpy as jnp
from form_eoms.form_and_solve import iterate

m = 10
k = 7


def lagrangian(q, dq, t):
    jax.debug.print("q {}; dq {}; t {}", q, dq, t)
    return 0.5 * 7 * jnp.dot(q, q) - 0.5 * jnp.dot(dq, dq)


a = iterate(
    q0=jnp.array([10.3]),
    r=1,
    lagrangian=lagrangian,
    t_sample_count=100,
    dt=2,
    t0=0
)
