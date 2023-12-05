import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from original import slimplectic
import slimpletic as st

m = 1.0
k = 1.0
ll = 1e-4 * np.sqrt(m * k)  # ll is $\lambda$ in the paper

dho = slimplectic.GalerkinGaussLobatto('t', ['q'], ['v'])
L = 0.5 * m * np.dot(dho.v, dho.v) - 0.5 * k * np.dot(dho.q, dho.q)
K = -ll * np.dot(dho.vp, dho.qm)


def lagrangian_f(q, qdot, t):
    return 0.5 * m * jnp.dot(qdot, qdot) - 0.5 * k * jnp.dot(q, q) ** 2


dho.discretize(L, K, 0, method='implicit', verbose=True)

dt = 0.1 * np.sqrt(m / k)
t_sample_count = 500
tmax = t_sample_count * np.sqrt(m / k)
t = dt * np.arange(0, int(tmax / dt) + 1)

q0 = [1.]
pi0 = [0.25 * dt * k]

q_slim2, pi_slim2 = dho.integrate(q0, pi0, t)

plt.plot(q_slim2[0], pi_slim2[0])
plt.show()

my_q, my_qdot = st.iterate(
    pi0=jnp.array(pi0),
    q0=jnp.array(q0),
    t0=0,
    dt=dt,
    t_sample_count=t_sample_count,
    r=0,
    lagrangian=lagrangian_f
)

plt.plot(my_q, my_qdot)
plt.show()
