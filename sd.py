import jax
from jaxopt import GaussNewton
from jax import numpy as jnp


def rosenbrock(x, additional_arg):
    a = additional_arg * x[0]
    b = x[1]

    return jnp.array([
        10 * (6 * b ** 4 - a ** 2),
        1 + 2 * a
    ])


additional_arg_value = 10.0

gn = GaussNewton(residual_fun=rosenbrock)
gn_sol = gn.run(jnp.array([1.0, 2.0]), additional_arg=additional_arg_value).params
jax.debug.print(
    "sol = {} with f(sol, {}) = {}",
    gn_sol,
    additional_arg_value,
    rosenbrock(gn_sol, additional_arg=additional_arg_value)
)
