# NOTE: THIS IS IMPORTANT
# Else the values will not agree with the original code.
from jax import config
config.update("jax_enable_x64", True)

from ggl import ggl
from original.slimplectic_GGL import GGLdefs
import jax.numpy as jnp


def floatify1(xs):
    return jnp.array([float(x) for x in xs])


def floatify2(dij):
    return jnp.array([[float(x) for x in row] for row in dij])


def test_ggl():
    for r in range(0, 20):
        original_xs, original_ws, original_dij = GGLdefs(r)
        new_xs, new_ws, new_dij = ggl(r)

        assert jnp.allclose(floatify1(original_xs), new_xs)
        assert jnp.allclose(floatify1(original_ws), new_ws)
        assert jnp.allclose(floatify2(original_dij), new_dij)
