from neural_networks.data import lookup_family
from neural_networks.data.generate_data_impl import setup_solver

import jax.numpy as jnp
import matplotlib.pyplot as plt

t0 = 0
dt = 0.1
t = t0 + dt * jnp.arange(0, 100 + 1)

solver = setup_solver(
    family=lookup_family('dho'),
    iterations=100,
)

emb1 = jnp.array([1.0, 1.0, 1.0])

q1, pi1 = solver(
    jnp.array([1.0, 1.0, 1.0]),
    jnp.array([0.0]),
    jnp.array([1.0]),
)

q10, pi10 = solver(
    10 * jnp.array([1.0, 1.0, 1.0]),
    jnp.array([0.0]),
    jnp.array([1.0]),
)

# %%

plt.plot(t, q1, label='q1')
plt.plot(t, q10 * 10, label='q10', linestyle='--')
plt.legend()
plt.show()

