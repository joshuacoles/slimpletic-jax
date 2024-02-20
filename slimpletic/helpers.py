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


def fill_out_initial(initial, r):
    # TODO: Replace r with generic count parameter
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
