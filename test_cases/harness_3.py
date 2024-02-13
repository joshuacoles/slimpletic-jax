import matplotlib.pyplot as plt

from harness import original, solver, dt, q0, pi0, t0, t_sample_count, t
from jax import numpy as jnp
import numpy as np

jax_results = solver.integrate_manual(jnp.array(q0), jnp.array(pi0), t0, t_sample_count)
original_results = original.integrate(
    q0=np.array(q0),
    pi0=np.array(pi0),
    t=t,
    dt=dt,
)

plt.plot(t, jax_results[0], label='JAX')
plt.plot(t, original_results[0], label='Original')
plt.title('q')
plt.show()

plt.plot(t, jax_results[1], label='JAX')
plt.plot(t, original_results[1], label='Original')
plt.title('pi')
plt.show()
