# %%
from pathlib import Path

import numpy as np

from figures import figures_root
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

l = 1.0
m = 1.0
g = 9.81


def cartesian(q):
    theta_1, theta_2 = q

    x_1 = l / 2 * jnp.sin(theta_1)
    y_1 = -l / 2 * jnp.cos(theta_1)

    x_2 = l * jnp.sin(theta_1) + l / 2 * jnp.sin(theta_2)
    y_2 = -l * jnp.cos(theta_1) - l / 2 * jnp.cos(theta_2)

    return jnp.array([x_1, y_1]), jnp.array([x_2, y_2])


def lagrangian(q, v, t):
    theta_1, theta_2 = q
    omega_1, omega_2 = v
    xy1, xy2 = cartesian(q)
    x_1, y_1 = xy1
    x_2, y_2 = xy2

    x_1_dot = l / 2 * jnp.cos(theta_1) * omega_1
    y_1_dot = l / 2 * jnp.sin(theta_1) * omega_1

    x_2_dot = l * jnp.cos(theta_1) * omega_1 + l / 2 * jnp.cos(theta_2) * omega_2
    y_2_dot = l * jnp.sin(theta_1) * omega_1 + l / 2 * jnp.sin(theta_2) * omega_2

    moment_of_inertia = 1 / 12 * m * l ** 2

    linear_kinetic = 1 / 2 * m * (x_1_dot ** 2 + y_1_dot ** 2 + x_2_dot ** 2 + y_2_dot ** 2)
    angular_kinetic = moment_of_inertia / 2 * (omega_1 ** 2 + omega_2 ** 2)
    potential = m * g * (y_1 + y_2)

    return linear_kinetic + angular_kinetic - potential


def nonconservative(qp, qm, vp, vm, t):
    # This is damping in the first pendulum
    return -2 * t * vp[0] * qm[0]


dt = 0.01
r = 2
ggl_bundle = GGLBundle(r=r)

solver = SolverScan(DiscretisedSystem(
    dt=dt,
    ggl_bundle=ggl_bundle,
    lagrangian=lagrangian,
    k_potential=nonconservative
))


def wrapped_solve(q0, pi0):
    print(q0.shape, pi0.shape)
    return solver.integrate(
        q0=q0,
        pi0=pi0,
        t0=0,
        iterations=iterations,
        additional_data=None,
        result_orientation='coordinate'
    )


vmapped_solve = jax.vmap(wrapped_solve, in_axes=(0, 0))

# %%

omega_1 = jnp.array([-4, -2, 0.0, 2, 4])
omega_2 = jnp.arange(-2, 2, 0.1)

pi0 = jnp.array(jnp.meshgrid(
    omega_1,
    omega_2
)).T.reshape(-1, 2)

q0 = jnp.repeat(jnp.array([0.0, 0.0], dtype=jnp.float64)[jnp.newaxis, :], axis=0, repeats=pi0.shape[0])

iterations = 10_000
t = solver.time_domain(t0=0, iterations=iterations)

q, pi = vmapped_solve(
    q0,
    pi0,
)

# %%

t_max = 100

for sys_i in range(q.shape[0]):
    qq = q[sys_i]
    modded_data = jnp.mod(qq[1, :t_max], jnp.pi * 2)
    data = jax.lax.select(modded_data > jnp.pi, modded_data - 2 * jnp.pi, modded_data)
    abs_d_data = jnp.abs(jnp.diff(data))
    mask = jnp.hstack([abs_d_data > abs_d_data.mean() + 3 * abs_d_data.std(), [False]])
    masked_data = np.ma.MaskedArray(data, mask)
    plt.plot(qq[0, :t_max], masked_data)

plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.xticks([-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi], ['-$\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
plt.yticks([-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi], ['-$\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
plt.savefig('out.pdf')
plt.show()
