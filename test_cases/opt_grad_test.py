from jaxopt import GaussNewton
from jax import numpy as jnp, grad, jit

a = jnp.array([1.0, 1.0])
newton = GaussNewton(
    residual_fun=lambda x, p: jnp.dot(p, (x ** 2 - 1.0) * a)
)


def fn(p):
    return newton.run(jnp.array(1.0), p).params


print(grad(fn)(jnp.array([1.0, 1.0])))
