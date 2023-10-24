import jax
from jaxopt import GaussNewton
from jax import numpy as jnp


def rosenbrock(x):
    a = x[0]
    b = x[1]

    return jnp.array([
        10 * (6 * b ** 4 - a ** 2),
        1 + 2 * a
    ])


gn = GaussNewton(residual_fun=rosenbrock)
gn_sol = gn.run(jnp.array([1.0, 2.0])).params
jax.debug.print("sol = {} with f(sol) = {}", gn_sol, rosenbrock(gn_sol))
