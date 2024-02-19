import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad
from slimpletic import Solver


@jit
def compute_action(state, embedding):
    dof = state.size
    return jax.lax.fori_loop(
        0, dof ** 2,
        lambda i, acc: acc + (embedding[i] * state[i // dof] * state[i % dof]),
        0.0
    )


# print(compute_action(jnp.array(np.random.rand(3)), jnp.array(np.random.rand(9))))
# print(grad(compute_action, argnums=(1,))(jnp.array(np.random.rand(3)), jnp.array(np.random.rand(9))))

embedding = jnp.array(np.random.rand(9)) * 10

print(embedding)

solver = Solver(
    dt=0.1,
    r=0,
    lagrangian=jit(lambda q, v, t: compute_action(jnp.concat([q, v], axis=0), embedding)),
    k_potential=None
)

print(solver.integrate(
    q0=jnp.array([1.0, 0.0, 0.0]),
    pi0=jnp.array([0.0, 0.0, 0.0]),
    t0=0.0,
    iterations=10
))
