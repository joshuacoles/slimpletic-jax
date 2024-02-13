import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from sympy import Symbol

from slimpletic import Solver
from original import GalerkinGaussLobatto

# System parameters, used in both methods
m = 1.0
k = 100.0
ll = 1e-4 * np.sqrt(m * k)  # ll is $\lambda$ in the paper

# Simulation and Method parameters
dt = 0.1 * np.sqrt(m / k)
t_sample_count = 10
tmax = t_sample_count * np.sqrt(m / k)
t0 = 1
t = t0 + dt * np.arange(0, t_sample_count + 1)
r = 0

# Initial conditions
q0 = [1.]
pi0 = [0.25 * dt * k]

# Original method
original = GalerkinGaussLobatto('t', ['q'], ['v'])
L = 0.5 * m * np.dot(original.v, original.v) - 0.5 * k * np.dot(original.q, original.q)

# DHO:
K = -ll * np.dot(original.vp, original.qm)
# No damping:
K_nd = Symbol('a')
original.discretize(L, K_nd, r, method='implicit', verbose=False)


# JAX Method
def lagrangian_f(q, v, t):
    return 0.5 * m * jnp.dot(v, v) - 0.5 * k * jnp.dot(q, q)


solver = Solver(r=r, dt=dt, lagrangian=lagrangian_f)

# jax_results = solver.integrate(jnp.array(q0), jnp.array(pi0), t0, t_sample_count)
# original_results = original.integrate(
#     q0=np.array(q0),
#     pi0=np.array(pi0),
#     t=t,
#     dt=dt,
# )
