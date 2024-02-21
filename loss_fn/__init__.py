from functools import partial

import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad
from slimpletic import DiscretisedSystem, SolverScan, GGLBundle


@jit
def compute_action(state, embedding):
    dof = state.size
    return jax.lax.fori_loop(
        0, dof ** 2,
        lambda i, acc: acc + (embedding[i] * state[i // dof] * state[i % dof]),
        0.0
    )


def embedded_lagrangian(q, v, t, embedding):
    return compute_action(jnp.concat([q, v], axis=0), embedding)


ggl_bundle = GGLBundle(r=0)

# The system which will be used when computing the loss function.
test_system_solver = SolverScan(DiscretisedSystem(
    dt=0.1,
    ggl_bundle=ggl_bundle,
    lagrangian=embedded_lagrangian,
    k_potential=None,
    pass_additional_data=True
))


def rms(x, y):
    return jnp.sqrt(jnp.mean((x - y) ** 2))


def loss_fn(embedding: jnp.ndarray, target_q: jnp.ndarray, target_pi: jnp.ndarray):
    assert target_q.size == target_pi.size

    q, pi = test_system_solver.integrate(
        q0=jnp.array([1.0, 0.0, 0.0]),
        pi0=jnp.array([0.0, 0.0, 0.0]),
        t0=0,
        iterations=10,
        additional_data=embedding
    )

    return jnp.sqrt(rms(q, target_q) ** 2 + rms(pi, target_pi) ** 2)


expected_system_solver = SolverScan(DiscretisedSystem(
    dt=0.1,
    ggl_bundle=ggl_bundle,
    lagrangian=lambda q, v, t: jnp.dot(q, q) + jnp.dot(v, v),
    k_potential=None,
))

exptected_q, expected_pi = expected_system_solver.integrate(
    q0=jnp.array([1.0, 0.0, 0.0]),
    pi0=jnp.array([0.0, 0.0, 0.0]),
    t0=0.0,
    iterations=10
)

print(grad(loss_fn, argnums=(0,))(
    jnp.array(np.random.rand(9)),
    exptected_q,
    expected_pi
))

# loss_fn(
#     jnp.array(np.random.rand(9)),
#     exptected_q,
#     expected_pi
# )
