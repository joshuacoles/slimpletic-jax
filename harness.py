import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax

from slimpletic import Solver

# System parameters, used in both methods
m = 1.0
k = 100.0
ll = 1e-4 * np.sqrt(m * k)  # ll is $\lambda$ in the paper

# Simulation and Method parameters
dt = 0.1 * np.sqrt(m / k)
t_sample_count = 1
tmax = t_sample_count * np.sqrt(m / k)
t0 = 0
t = t0 + dt * np.arange(0, t_sample_count + 1)
r = 0

# Initial conditions
q0 = [1.]
pi0 = [0.25 * dt * k]


def lagrangian_f(q, qdot, t):
    return 0.5 * m * jnp.dot(qdot, qdot) - 0.5 * k * jnp.dot(q, q)


solver = Solver(r=r, dt=dt, lagrangian=lagrangian_f)

solver.integrate_manual(jnp.array(q0), jnp.array(pi0), t0, t_sample_count)
