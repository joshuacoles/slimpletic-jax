from functools import partial

import jax.lax
import jax.numpy as jnp
import jaxopt
import numpy as np
from jax import jit, grad
from matplotlib import pyplot as plt

from slimpletic import DiscretisedSystem, SolverScan, GGLBundle

q0 = jnp.array([0.0])
pi0 = jnp.array([1.0])
t0 = 0
iterations = 100
dt = 0.1
dof = 1

ggl_bundle = GGLBundle(r=0)


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


# The system which will be used when computing the loss function.
test_system_solver = SolverScan(DiscretisedSystem(
    dt=dt,
    ggl_bundle=ggl_bundle,
    lagrangian=embedded_lagrangian,
    k_potential=None,
    pass_additional_data=True
))


def rms(x, y):
    return jnp.sqrt(jnp.mean((x - y) ** 2))


@partial(jit, static_argnums=(1, 2))
def loss_fn(embedding: jnp.ndarray, target_q: jnp.ndarray, target_pi: jnp.ndarray):
    q, pi = test_system_solver.integrate(
        q0=q0,
        pi0=pi0,
        t0=t0,
        iterations=iterations,
        additional_data=embedding
    )

    return jnp.sqrt(rms(q, target_q) ** 2 + rms(pi, target_pi) ** 2)


expected_system_solver = SolverScan(DiscretisedSystem(
    dt=dt,
    ggl_bundle=ggl_bundle,
    lagrangian=lambda q, v, t: 0.5 * jnp.dot(v, v) - 0.5 * jnp.dot(q, q),
    k_potential=None,
))

exptected_q, expected_pi = expected_system_solver.integrate(
    q0=q0,
    pi0=pi0,
    t0=t0,
    iterations=iterations,
)

results = jaxopt.GradientDescent(
    loss_fn,
    maxiter=1000,
    verbose=True,
).run(
    jnp.array(np.random.rand(dof ** 2)),
    exptected_q,
    expected_pi
).params

sol_q, sol_pi = test_system_solver.integrate(
    q0=q0,
    pi0=pi0,
    t0=t0,
    iterations=iterations,
    additional_data=results
)

t = t0 + dt * np.arange(0, iterations + 1)
plt.plot(t, exptected_q)
plt.plot(t, sol_q, linestyle='dashed')
plt.show()
