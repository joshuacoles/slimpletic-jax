import time

import matplotlib.pyplot as plt
import jax.numpy as jnp

import original
from slimpletic import make_solver
import numpy as np
from rk import RungeKutta2, RungeKutta4

# Set harmonic oscillator parameters
m = 1.0
k = 1.0
ll = 1e-4 * np.sqrt(m * k)  # ll is $\lambda$ in the paper


def lagrangian(q, v, t):
    """Simple damped harmonic oscillator Lagrangian"""
    return 0.5 * m * jnp.dot(v, v) - 0.5 * k * jnp.dot(q, q)


def nonconservative(qp, qm, vp, vm, t):
    """Nonconservative part of the Lagrangian"""
    return -ll * jnp.dot(vp, qm)


# Time samples
dt = 0.1 * np.sqrt(m / k)
tmax = 10000 * np.sqrt(m / k)
t = dt * np.arange(0, int(tmax / dt) + 1)

# Initial data (at t=0)
q0 = jnp.array([1.])

# The initial condition for pi0 is chosen because the 2nd order slimplectic method
# has $\pi$ actually evaluated at the mid-step, and it needs corrections to that effect.
# Otherwise, the phase is off and the energy has a constant offset.
pi0_order2 = jnp.array([0.25 * dt * k])
pi0_order4 = jnp.array([0.])

print("Compute updated implementation...")

# region Updated Implementation

# Create an instance of the GalerkinGaussLobatto class and call it `dho` for damped harmonic oscillator.
dho_2 = make_solver(
    r=0,
    dt=dt,
    lagrangian=lagrangian, k_potential=nonconservative
)

dho_4 = make_solver(
    r=1,
    dt=dt,
    lagrangian=lagrangian, k_potential=nonconservative
)

# Updated Implementation
q_slim2, pi_slim2 = dho_2.integrate(q0, pi0_order2, t0=0, iterations=t.size - 1, result_orientation='coordinate')
q_slim4, pi_slim4 = dho_4.integrate(q0, pi0_order4, t0=0, iterations=t.size - 1, result_orientation='coordinate')

# The integrator returns things as a list of arrays, so we need to reshape them to be 1D arrays.
q_slim2 = q_slim2.reshape(-1)
pi_slim2 = pi_slim2.reshape(-1)
q_slim4 = q_slim4.reshape(-1)
pi_slim4 = pi_slim4.reshape(-1)

# endregion

print("Compute Runge-Kutta...")

# region Runge-Kutta
# Runge-Kutta integrators
rk2 = RungeKutta2()
rk4 = RungeKutta4()


# Define the derivative operator for a simple damped harmonic oscillator
def dydt(time, y):
    deriv = np.zeros(2)
    [q_x, v_x] = y
    deriv[0] = v_x
    deriv[1] = - (k / m) * q_x - (ll / m) * v_x
    return deriv


# Integrate
v0 = [0.]
q_rk2, v_rk2 = rk2.integrate(q0, v0, t, dydt)
q_rk4, v_rk4 = rk4.integrate(q0, v0, t, dydt)

# endregion

print("Compute original implementation...")

# region Original Implementation

original_2 = original.dho(m, k, ll, r=0)
original_4 = original.dho(m, k, ll, r=1)

print("Original 2")

q_org2, pi_org2, _ = original_2(
    t=t,
    q0=q0.tolist(),
    pi0=pi0_order2.tolist()
)

print("Original 4")

q_org4, pi_org4, _ = original_4(
    t=t,
    q0=q0.tolist(),
    pi0=pi0_order2.tolist()
)

# endregion

# Analytical solution
Omega = np.sqrt(k / m - ll ** 2 / 4.)
phi0 = - np.arctan(-ll / (2. * Omega))


def analytic_q(time):
    """Analytical solution for simple damped harmonic oscillator amplitude with q0=1, v0=0"""
    return np.exp(-ll * time / 2.) * np.cos(Omega * time + phi0)


def analytic_v(time):
    """Analytical solution for simple damped harmonic oscillator velocity with q0=1, v0=0"""
    return np.exp(-ll * time / 2.) * (-ll / 2. * np.cos(Omega * time + phi0) - Omega * np.sin(Omega * time + phi0))


def Energy(q, v):
    return 0.5 * m * v ** 2 + 0.5 * k * q ** 2


print("Computing energies...")

# Energies from the analytic solution and from different integrators
analytic_energy = Energy(analytic_q(t), analytic_v(t))
original_2_energy = Energy(q_org2, pi_org2 / m)
original_4_energy = Energy(q_org4, pi_org4 / m)
updated_order4_energy = Energy(q_slim4, pi_slim4 / m)
updated_order2_energy = Energy(q_slim2, pi_slim2 / m)
rk2_energy = Energy(q_rk2[0], v_rk2[0])
rk4_energy = Energy(q_rk4[0], v_rk4[0])

original_2_ef = jnp.abs(original_2_energy / analytic_energy - 1.)
original_4_ef = jnp.abs(original_4_energy / analytic_energy - 1.)
updated_order2_ef = jnp.abs(updated_order2_energy / analytic_energy - 1.)
updated_order4_ef = jnp.abs(updated_order4_energy / analytic_energy - 1.)
rk2_ef = jnp.abs(rk2_energy / analytic_energy - 1.)
rk4_ef = jnp.abs(rk4_energy / analytic_energy - 1.)

fig2 = plt.figure(figsize=(12, 5), dpi=500)

ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_ylim(1e-10, 1e1)
ax2.set_xlim(0.1, 10000)

ax2.loglog(t, original_2_ef, 'r-', linewidth=2.0, rasterized=True)
ax2.loglog(t, original_4_ef, 'r-', linewidth=2.0, rasterized=True)
ax2.loglog(t, updated_order2_ef, 'r-', linewidth=2.0, rasterized=True)
ax2.loglog(t, updated_order4_ef, color='orange', linestyle='-', linewidth=2.0, rasterized=True)
ax2.loglog(t, rk2_ef, 'g--', linewidth=2.0, rasterized=True)
ax2.loglog(t, rk4_ef, 'b--', linewidth=2.0, rasterized=True)

ax2.set_xlabel('Time, $t$ [$(m/k)^{1/2}$]', fontsize=18)
ax2.set_ylabel('Fractional energy error, $\delta E/E$', fontsize=18)

ax2.tick_params(axis='both', which='major', labelsize=16)
fig2.show()
fig2.savefig(f"{time.time()}-dho.pdf")
