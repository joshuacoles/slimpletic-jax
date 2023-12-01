from jax import numpy as jnp


def floatify1(xs):
    return jnp.array([float(x) for x in xs])


def floatify2(dij):
    return jnp.array([[float(x) for x in row] for row in dij])


# NOTE: THIS IS IMPORTANT
# Else the values will not agree with the original code.
# This should be called ASAP when running any code that uses jax.
def jax_enable_x64():
    from jax import config
    config.update("jax_enable_x64", True)
