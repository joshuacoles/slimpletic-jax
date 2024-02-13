import matplotlib.pyplot as plt

from harness import original, solver, dt, q0, pi0, t0, t_sample_count, t
from jax import numpy as jnp
import numpy as np

jax_results = solver.integrate(jnp.array(q0), jnp.array(pi0), t0, t_sample_count)
original_results = original.integrate(
    q0=np.array(q0),
    pi0=np.array(pi0),
    t=t,
    dt=dt,
)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Define a subplot with 1 row and 2 columns

# Plot your first chart on the first subplot
axs[0].plot(t, original_results[0], label='Original')
axs[0].plot(t, jax_results[0], label='JAX', linestyle='dashed')
axs[0].legend()
axs[0].set_title('q')

# Plot your second chart on the second subplot
axs[1].plot(t, original_results[1], label='Original')
axs[1].plot(t, jax_results[1], label='JAX', linestyle='dashed')
axs[1].legend()
axs[1].set_title('pi')

plt.show()
