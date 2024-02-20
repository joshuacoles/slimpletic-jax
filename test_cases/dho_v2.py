import random
import time

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from original import GalerkinGaussLobatto

# System parameters, used in both methods
m = 1.0
k = 1.0
ll = 0.5 * np.sqrt(m * k)  # ll is $\lambda$ in the paper

# Simulation and Method parameters
dt = 0.1 * np.sqrt(m / k)
iterations = 10_000
tmax = iterations * np.sqrt(m / k)
t0 = 1
t = t0 + dt * np.arange(0, iterations + 1)
r = 0

# Initial conditions
q0 = [1.]
pi0 = [0.25 * dt * k]

# Original method
original = GalerkinGaussLobatto('t', ['q'], ['v'])
L = 0.5 * m * np.dot(original.v, original.v) - 0.5 * k * np.dot(original.q, original.q)

# DHO:
K = -ll * np.dot(original.vp, original.qm)
original.discretize(L, K, r)


# JAX Method
def lagrangian_f(q, v, t):
    return 0.5 * m * jnp.dot(v, v) - 0.5 * k * jnp.dot(q, q)


def k_potential_f(qp, qm, vp, vm, t):
    return -ll * jnp.dot(vp, qm)


from slimpletic.v2_interface import DiscretisedSystem, GGLBundle, SolverBatchedScan, SolverScan

system = DiscretisedSystem(
    ggl_bundle=GGLBundle(r=r),
    dt=dt,
    lagrangian=lagrangian_f,
    k_potential=k_potential_f,
)

# solver = SolverScan(system)
solver = SolverBatchedScan(system, batch_size=100)

# solver = Solver(r=r, dt=dt, lagrangian=lagrangian_f, k_potential=k_potential_f)
dof = original.degrees_of_freedom

q0 = jnp.array(q0)
pi0 = jnp.array(pi0)

jax_start_time = time.time()
jax_q, jax_pi = solver.integrate(q0, pi0, t0, iterations=iterations)
jax_end_time = time.time()
jax_time = jax_end_time - jax_start_time
print(f"JAX time: {jax_time}")

original_start_time = time.time()
original_results = original.integrate(
    q0=np.array(q0),
    pi0=np.array(pi0),
    t=t,
    dt=dt,
)

original_end_time = time.time()
original_time = original_end_time - original_start_time
print(f"Original time: {original_time}")

plt.plot(t, jax_q)
plt.plot(t, original_results[0], linestyle='dashed')
plt.show()
