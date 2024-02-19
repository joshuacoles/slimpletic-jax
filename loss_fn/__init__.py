import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad


@jit
def compute_action(state, embedding):
    dof = state.size
    return jax.lax.fori_loop(
        0, dof ** 2,
        lambda i, acc: acc + (embedding[i] * state[i // dof] * state[i % dof]),
        0.0
    )


print(compute_action(jnp.array(np.random.rand(3)), jnp.array(np.random.rand(9))))
print(grad(compute_action, argnums=(1,))(jnp.array(np.random.rand(3)), jnp.array(np.random.rand(9))))
