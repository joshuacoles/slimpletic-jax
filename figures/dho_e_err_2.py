import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
import jax.numpy as jnp

import original
from slimpletic import make_solver
import numpy as np
from figures.rk import RungeKutta2, RungeKutta4
import pandas as pd

# %%

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

# %%

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

# %%

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

# %%
original_2 = original.dho(m, k, ll, r=0)

q_org2, pi_org2, _ = original_2(
    t=t,
    q0=q0.tolist(),
    pi0=pi0_order2.tolist()
)

original_4 = original.dho(m, k, ll, r=1)

q_org4, pi_org4, _ = original_4(
    t=t,
    q0=q0.tolist(),
    pi0=pi0_order4.tolist()
)

# %%
Omega = np.sqrt(k / m - ll ** 2 / 4.)
phi0 = - np.arctan(-ll / (2. * Omega))


def analytic_q(time):
    """Analytical solution for simple damped harmonic oscillator amplitude with q0=1, v0=0"""
    return np.exp(-ll * time / 2.) * np.cos(Omega * time + phi0)


def analytic_v(time):
    """Analytical solution for simple damped harmonic oscillator velocity with q0=1, v0=0"""
    return np.exp(-ll * time / 2.) * (-ll / 2. * np.cos(Omega * time + phi0) - Omega * np.sin(Omega * time + phi0))


# %%

pd.DataFrame({'q_org4': q_org4.reshape(-1), 'q_org2': q_org2.reshape(-1), 'pi_org2': pi_org2.reshape(-1),
              'pi_org4': pi_org4.reshape(-1)}).to_csv('original_e_err.csv')

pd.DataFrame({'q_slim4': q_slim4, 'q_slim2': q_slim2, 'pi_slim2': pi_slim2, 'pi_slim4': pi_slim4}).to_csv(
    'slimplectic_e_err.csv')

pd.DataFrame({'q_rk4': q_rk4.reshape(-1), 'q_rk2': q_rk2.reshape(-1), 'v_rk2': v_rk2.reshape(-1),
              'v_rk4': v_rk4.reshape(-1)}).to_csv('rk_e_err.csv')


# %%

def Energy(q, v):
    return 0.5 * m * v ** 2 + 0.5 * k * q ** 2


# Energies from the analytic solution and from different integrators
analytic_energy = Energy(analytic_q(t), analytic_v(t))
original_2_energy = Energy(q_org2, pi_org2 / m)
original_4_energy = Energy(q_org4, pi_org4 / m)
updated_order4_energy = Energy(q_slim4, pi_slim4 / m)
updated_order2_energy = Energy(q_slim2, pi_slim2 / m)
rk2_energy = Energy(q_rk2[0], v_rk2[0])
rk4_energy = Energy(q_rk4[0], v_rk4[0])

energies = pd.DataFrame({
    'analytic_energy': analytic_energy.reshape(-1),
    'original_2_energy': original_2_energy.reshape(-1),
    'original_4_energy': original_4_energy.reshape(-1),
    'updated_order4_energy': updated_order4_energy.reshape(-1),
    'updated_order2_energy': updated_order2_energy.reshape(-1),
    'rk2_energy': rk2_energy.reshape(-1),
    'rk4_energy': rk4_energy.reshape(-1)
})

energies.to_csv('energies.csv')

# %%

energies = pd.read_csv('energies.csv')

original_2_ef = np.abs(energies.original_2_energy / energies.analytic_energy - 1.)
original_4_ef = np.abs(energies.original_4_energy / energies.analytic_energy - 1.)
updated_order2_ef = np.abs(energies.updated_order2_energy / energies.analytic_energy - 1.)
updated_order4_ef = np.abs(energies.updated_order4_energy / energies.analytic_energy - 1.)
rk2_ef = np.abs(energies.rk2_energy / energies.analytic_energy - 1.)
rk4_ef = np.abs(energies.rk4_energy / energies.analytic_energy - 1.)

# %%

rk = pd.read_csv('rk_e_err.csv')
slim = pd.read_csv('slimplectic_e_err.csv')
org = pd.read_csv('original_e_err.csv')

analytic_momentum = m * analytic_v(t)
original_2_momentum = org.pi_org2
original_4_momentum = org.pi_org4
updated_order4_momentum = slim.pi_slim4
updated_order2_momentum = slim.pi_slim2
rk2_momentum = rk.v_rk2 * m
rk4_momentum = rk.v_rk4 * m

momenta = pd.DataFrame({
    'analytic_momentum': analytic_momentum,
    'original_2_momentum': original_2_momentum,
    'original_4_momentum': original_4_momentum,
    'updated_order4_momentum': updated_order4_momentum,
    'updated_order2_momentum': updated_order2_momentum,
    'rk2_momentum': rk2_momentum,
    'rk4_momentum': rk4_momentum
})

momenta.to_csv('momenta.csv')

# %%

momenta = pd.read_csv('momenta.csv')

fractional_momenta_updated_order2 = np.abs(momenta.updated_order2_momentum / momenta.analytic_momentum - 1.)
fractional_momenta_updated_order4 = np.abs(momenta.updated_order4_momentum / momenta.analytic_momentum - 1.)
fractional_momenta_rk2 = np.abs(momenta.rk2_momentum / momenta.analytic_momentum - 1.)
fractional_momenta_rk4 = np.abs(momenta.rk4_momentum / momenta.analytic_momentum - 1.)

# %%

# Momentum Plot

fig3 = plt.figure(figsize=(12, 5), dpi=500)
ax3 = fig3.add_subplot(2, 1, 2)

ax3.set_ylim(1e-12, 1e5)
ax3.set_xlim(0.1, 10000)


# fractional_momenta_updated_order2_min, fractional_momenta_updated_order2_max  = np.minimum.accumulate(fractional_momenta_updated_order2), np.maximum.accumulate(fractional_momenta_updated_order2)
# fractional_momenta_updated_order4_min, fractional_momenta_updated_order4_max  = np.minimum.accumulate(fractional_momenta_updated_order4), np.maximum.accumulate(fractional_momenta_updated_order4)
# fractional_momenta_rk2_min, fractional_momenta_rk2_max  = np.minimum.accumulate(fractional_momenta_rk2), np.maximum.accumulate(fractional_momenta_rk2)
# fractional_momenta_rk4_min, fractional_momenta_rk4_max  = np.minimum.accumulate(fractional_momenta_rk4), np.maximum.accumulate(fractional_momenta_rk4)

# ax3.fill_between(t, fractional_momenta_updated_order2_min, fractional_momenta_updated_order2_max, color='red', alpha=0.5, label='Updated 2nd order')
# ax3.fill_between(t, fractional_momenta_updated_order4_min, fractional_momenta_updated_order4_max, color='orange', alpha=0.5, label='Updated 4th order')
# ax3.fill_between(t, fractional_momenta_rk2_min, fractional_momenta_rk2_max, color='green', alpha=0.5, label='Runge-Kutta 2')
# ax3.fill_between(t, fractional_momenta_rk4_min, fractional_momenta_rk4_max, color='blue', alpha=0.5, label='Runge-Kutta 4')


# ax3.plot(t, fractional_momenta_rk2, 'g--', linewidth=2.0, rasterized=True, label='Runge-Kutta 2')
# ax3.plot(t, fractional_momenta_rk4, 'b--', linewidth=2.0, rasterized=True, label='Runge-Kutta 4')

ax3.set_xlabel('Time, $t$ [$(m/k)^{1/2}$]', fontsize=18)
ax3.set_ylabel('Fractional momenta error, $\delta E/E$', fontsize=18)

ax3.set_xscale('log')
ax3.set_yscale('log')


ax3.tick_params(axis='both', which='major', labelsize=16)
fig3.legend()
fig3.show()

# %%

fig2 = plt.figure(figsize=(12, 5), dpi=500)
fig2.subplots_adjust(hspace=0.5, bottom=0.2)

ax2 = fig2.add_subplot(2, 1, 1)
ax3 = fig2.add_subplot(2, 1, 2)
ax2.set_ylim(1e-10, 1e1)
ax2.set_xlim(0.1, 10000)

ax2.loglog(t, updated_order2_ef, 'r-', linewidth=2.0, label='Updated 2nd order')
ax2.loglog(t, updated_order4_ef, color='orange', linestyle='-', linewidth=2.0, label='Updated 4th order')
ax2.loglog(t, rk2_ef, 'g--', linewidth=2.0, rasterized=True, label='Runge-Kutta 2')
ax2.loglog(t, rk4_ef, 'b--', linewidth=2.0, rasterized=True, label='Runge-Kutta 4')

# ax2.set_xlabel('Time, $t$ [$(m/k)^{1/2}$]', fontsize=18)
ax2.set_ylabel('$\delta E/E$', fontsize=18)

ax2.tick_params(axis='both', which='major', labelsize=16)

# Momentum Plot

ax3.set_ylim(1e-12, 1e5)
ax3.set_xlim(0.1, 10000)

ax3.loglog(t, fractional_momenta_updated_order2, 'r-', linewidth=2.0)
ax3.loglog(t, fractional_momenta_updated_order4, color='orange', linestyle='-', linewidth=2.0)

ax3.set_xlabel('Time, $t$ [$(m/k)^{1/2}$]', fontsize=18)
ax3.set_ylabel('$\delta p/p$', fontsize=18)

ax3.tick_params(axis='both', which='major', labelsize=16)

fig2.legend()
fig2.show()

# %%
# Simple data to display in various forms
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

fig = plt.figure()
# set height ratios for subplots
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

# the first subplot
ax2 = plt.subplot(gs[0])

ax2.set_ylim(1e-10, 1e1)
ax2.set_xlim(0.1, 10000)

ax2.loglog(t, updated_order2_ef, 'r-', linewidth=2.0, label='Updated 2nd order')
ax2.loglog(t, updated_order4_ef, color='orange', linestyle='-', linewidth=2.0, label='Updated 4th order')
ax2.loglog(t, rk2_ef, 'g--', linewidth=2.0, rasterized=True, label='Runge-Kutta 2')
ax2.loglog(t, rk4_ef, 'b--', linewidth=2.0, rasterized=True, label='Runge-Kutta 4')


# the second subplot
# shared axis X
ax3 = plt.subplot(gs[1], sharex = ax2)

ax3.set_ylim(1e-12, 1e5)
ax3.set_xlim(0.1, 10000)

ax3.loglog(t, fractional_momenta_updated_order2, 'r-', linewidth=2.0)
ax3.loglog(t, fractional_momenta_updated_order4, color='orange', linestyle='-', linewidth=2.0)
plt.setp(ax3.get_xticklabels(), visible=False)
yticks = ax3.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax2.legend()

# remove vertical gap between subplots
plt.subplots_adjust(hspace=.0)
plt.show()
