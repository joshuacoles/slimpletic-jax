import jax
import jax.numpy as jnp
from jax import jit, Array

m = 1.0
k = 1.0


@jit
def L(q_vec, q_dot_vec, t):
    return 0.5 * m * jnp.dot(q_dot_vec, q_dot_vec) - 0.5 * k * jnp.dot(q_vec, q_vec)


jax.debug.print("L {}", L(
    jnp.array([1.0, 2.0]),
    jnp.array([1.0, 3.0]),
    1.0,
))


def take_vector_grad(f, i, arg='q'):
    if arg == 'q':
        def f_with_i_pulled_out(q_i, q_vec: Array, q_dot_vec, t):
            return f(q_vec.at[i].set(q_i), q_dot_vec, t)
    elif arg == 'q_dot':
        def f_with_i_pulled_out(q_dot_i, q_vec: Array, q_dot_vec, t):
            return f(q_vec, q_dot_vec.at[i].set(q_dot_i), t)
    else:
        raise ValueError(f"arg must be 'q' or 'q_dot', got {arg}")

    dfdi = jax.grad(jit(f_with_i_pulled_out), argnums=0)

    return jit(lambda q_vec, q_dot_vec, t: dfdi(q_vec[i], q_vec, q_dot_vec, t))


dl_dq = take_vector_grad(L, 0, arg='q')

jax.debug.print("L {}", dl_dq(
    jnp.array([1.0, 2.0]),
    jnp.array([1.0, 3.0]),
    1.0,
))
