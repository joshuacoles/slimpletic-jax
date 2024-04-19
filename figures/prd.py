from slimpletic import GGLBundle, DiscretisedSystem, SolverScan

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import figures.orbit_util as orbit
from figures.rk import RungeKutta2, RungeKutta4

# Parameters
G = 39.478758435  # (in AU^3/M_sun/yr^2))
M_Sun = 1.0  # (in solar masses)
rho = 2.0  # (in g/cm^3)
d = 5.0e-3  # (in cm)
beta = 0.0576906 * (2.0 / rho) * (1.0e-3 / d)  # (dimensionless)
c = 63241.3  # (in AU/yr)
m = 1.


def lagrangian(q, v, t):
    return 0.5 * jnp.dot(v, v) + (1.0 - beta) * G * M_Sun / jnp.dot(q, q) ** 0.5


def nonconservative(qp, qm, vp, vm, t):
    a = jnp.dot(vp, qm) + jnp.dot(vp, qp) * jnp.dot(qp, qm) / jnp.dot(qp, qp)
    b = -beta * G * M_Sun / c / jnp.dot(qp, qp)
    return a * b


# We take the initial orbital parameters to be given by:
# a=1, e=0, i=0, omega=0, Omega=0, M=0
q0, v0 = orbit.Calc_Cartesian(1.0, 0.2, 0.0, 0.0, 0.0, 0.0, (1.0 - beta) * G * M_Sun)
pi0 = v0  # Dust taken to have unit mass

# Time samples (in years)
t_end = 6000
dt = 0.005
t = np.arange(0, t_end + dt, dt)

# %%

ggl_r0 = GGLBundle(r=0)
ggl_r1 = GGLBundle(r=1)
ggl_r2 = GGLBundle(r=2)

solver_2 = SolverScan(DiscretisedSystem(
    dt=dt,
    ggl_bundle=ggl_r0,
    lagrangian=lagrangian,
    k_potential=nonconservative
))

solver_4 = SolverScan(DiscretisedSystem(
    dt=dt / 2,
    ggl_bundle=ggl_r1,
    lagrangian=lagrangian,
    k_potential=nonconservative
))

solver_6 = SolverScan(DiscretisedSystem(
    dt=dt,
    ggl_bundle=ggl_r2,
    lagrangian=lagrangian,
    k_potential=nonconservative
))

rk2 = RungeKutta2()
rk4 = RungeKutta4()

q_slim2, pi_slim2 = solver_2.integrate(
    q0[:2],
    pi0[:2],
    t0=t[0],
    iterations=t.size - 1,
    result_orientation='coordinate',
)

q_slim4, pi_slim4 = solver_4.integrate(
    q0[:2],
    pi0[:2],
    t0=t[0],
    iterations=t.size - 1,
    result_orientation='coordinate',
)

q_slim6, pi_slim6 = solver_6.integrate(
    q0[:2],
    pi0[:2],
    t0=t[0],
    iterations=t.size - 1,
    result_orientation='coordinate',
)


# Define the derivative operator
def dydt(time, y):
    deriv = np.zeros(4)
    [q_x, q_y, v_x, v_y] = y
    r = (q_x * q_x + q_y * q_y) ** 0.5
    deriv[0] = v_x
    deriv[1] = v_y
    deriv[2] = -(1. - beta) * G * M_Sun * q_x / (r * r * r)
    deriv[2] -= (beta * G * M_Sun / (c * r * r)) * (v_x + q_x * (q_x * v_x + q_y * v_y) / (r * r))
    deriv[3] = -(1. - beta) * G * M_Sun * q_y / (r * r * r)
    deriv[3] -= (beta * G * M_Sun / (c * r * r)) * (v_y + q_y * (q_x * v_x + q_y * v_y) / (r * r))

    return deriv


# Integrate
q_rk2, v_rk2 = rk2.integrate(q0[:2], v0[:2], t, dydt)
q_rk4, v_rk4 = rk4.integrate(q0[:2], v0[:2], t, dydt)


# Energy function
def compute_energy(q, v):
    return 0.5 * m * (v[0] ** 2 + v[1] ** 2) - (1. - beta) * G * M_Sun * m / np.sqrt(q[0] ** 2 + q[1] ** 2)


# Energies from different integrators
E_slim2 = compute_energy(q_slim2, pi_slim2 / m)
E_slim4 = compute_energy(q_slim4, pi_slim4 / m)
E_slim6 = compute_energy(q_slim6, pi_slim6 / m)

E_rk2 = compute_energy(q_rk2, v_rk2)
E_rk4 = compute_energy(q_rk4, v_rk4)

fig2 = plt.figure(figsize=(12, 5), dpi=500)
ax2 = fig2.add_subplot(1, 1, 1)

ax2.set_xlim(0.01, 6000)
ax2.loglog(t, np.abs(E_slim2 / E_slim6 - 1.), 'r-', linewidth=2.0, rasterized=True)
ax2.loglog(t, np.abs(E_slim4 / E_slim6 - 1.), '-', color='orange', linewidth=2.0, rasterized=True)
ax2.loglog(t, np.abs(E_rk2 / E_slim6 - 1.), 'g--', linewidth=2.0, rasterized=True)
ax2.loglog(t, np.abs(E_rk4 / E_slim6 - 1.), 'b--', linewidth=2.0, rasterized=True)

ax2.set_xlabel('Time, $t$ [yr]', fontsize=18)
ax2.set_ylabel('Fractional Energy Error, $\delta E/E_6$', fontsize=18)

ax2.text(15, 1.8e-4, r'$2^{nd}$ order Slimplectic', fontsize=15, color='black')
ax2.text(15, 0.5e-8, r'$4^{th}$ order Slimplectic', fontsize=15, color='black')
ax2.text(30, 1e-5, r'$4^{th}$ order RK', fontsize=15, color='blue', rotation=10)
ax2.text(30, 1.5e-1, r'$2^{nd}$ order RK', fontsize=15, color='g', rotation=10)
ax2.text(0.015, 1e-1, r'$\Delta t = 0.01$ yr', fontsize=18, color='black')

ax2.tick_params(axis='both', which='major', labelsize=16)

ax2.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1e0])
plt.show()
