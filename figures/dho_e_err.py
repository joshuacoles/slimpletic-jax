import time

import matplotlib.pyplot as plt
import jax.numpy as jnp
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

# Specify time samples at which the numerical solution is to be given initial data

# Time samples
dt = 0.1 * np.sqrt(m / k)
tmax = 10000 * np.sqrt(m / k)
t = dt * np.arange(0, int(tmax / dt) + 1)

# Initial data (at t=0)
q0 = jnp.array([1.])
pi0 = jnp.array([0.25 * dt * k])
# The initial condition for pi0 is chosen because the 2nd order slimplectic method 
# has $\pi$ actually evaluated at the mid-step, and it needs corrections to that effect.
# Otherwise, the phase is off and the energy has a constant offset. 

# Create an instance of the GalerkinGaussLobatto class and call it `dho` for damped harmonic oscillator.
dho_2 = make_solver(
    r=0,
    dt=dt,
    lagrangian=lagrangian, k_potential=nonconservative
)

# Now integrate the 2nd order slimplectic integrator
q_slim2, pi_slim2 = dho_2.integrate(q0, pi0, t0=0, iterations=t.size - 1)

# We can't mutate the solver instance, so we create a new one with similar parameters
dho_4 = make_solver(
    r=1,
    dt=dt,
    lagrangian=lagrangian, k_potential=nonconservative
)

q_slim4, pi_slim4 = dho_4.integrate(q0, pi0=jnp.array([0.]), t0=0, iterations=t.size - 1)

print("All done with the slimplectic integrators!")

# Instantiate the 2nd and 4th order Runge-Kutta classes
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

print("All done with the Runge-Kutta integrators!")

# Please note that q and pi are outputs of the slimplectic integration, 
# while q and v are output from the Runge-Kutta integrators.

# Analytical solution
Omega = np.sqrt(k / m - ll ** 2 / 4.)
phi0 = - np.arctan(-ll / (2. * Omega))


def q(time):
    """Analytical solution for simple damped harmonic oscillator amplitude with q0=1, v0=0"""
    return np.exp(-ll * time / 2.) * np.cos(Omega * time + phi0)


def v(time):
    """Analytical solution for simple damped harmonic oscillator velocity with q0=1, v0=0"""
    return np.exp(-ll * time / 2.) * (-ll / 2. * np.cos(Omega * time + phi0) - Omega * np.sin(Omega * time + phi0))

# Energy function
def Energy(q, v):
    return 0.5 * m * v ** 2 + 0.5 * k * q ** 2


# Energies from the analytic solution and from different integrators
E = Energy(q(t), v(t))

E_slim2 = Energy(q_slim2, pi_slim2 / m)
E_slim4 = Energy(q_slim4, pi_slim4 / m)

E_rk2 = Energy(q_rk2[0], v_rk2[0])
E_rk4 = Energy(q_rk4[0], v_rk4[0])

print("All done with the energy calculations!")

fig2 = plt.figure(figsize=(12, 5), dpi=500)

ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_ylim(1e-10, 1e1)
ax2.set_xlim(0.1, 10000)

print("All done with the plotting setup!")

e_slim2 = jnp.abs(E_slim2.reshape(E_slim2.shape[0]) / E - 1.)
e_slim4 = jnp.abs(E_slim4.reshape(E_slim4.shape[0]) / E - 1.)
e_rk2 = jnp.abs(E_rk2 / E - 1.)
e_rk4 = jnp.abs(E_rk4 / E - 1.)

print("All done with the error calculations!")

ax2.loglog(t, e_slim2, 'r-', linewidth=2.0, rasterized=True)
print("Plotted 1")
ax2.loglog(t, e_slim4, color='orange', linestyle='-', linewidth=2.0, rasterized=True)
print("Plotted 2")
ax2.loglog(t, e_rk2, 'g--', linewidth=2.0, rasterized=True)
print("Plotted 3")
ax2.loglog(t, e_rk4, 'b--', linewidth=2.0, rasterized=True)
print("Plotted 4")

ax2.set_xlabel('Time, $t$ [$(m/k)^{1/2}$]', fontsize=18)
ax2.set_ylabel('Fractional energy error, $\delta E/E$', fontsize=18)
print("Set labels")

ax2.text(0.12, 0.75, r'$\Delta t = 0.1 (m/k)^{1/2}$', fontsize=18, color='k')
ax2.text(50, 3e-4, r'$2^{nd}$ order Slimplectic', fontsize=15, color='k')
ax2.text(50, 6e-8, r'$4^{th}$ order Slimplectic', fontsize=15, color='black')
ax2.text(2, 3e-6, r'$4^{th}$ order RK', fontsize=15, color='b', rotation=11)
ax2.text(50, 1.5e-1, r'$2^{nd}$ order RK', fontsize=15, color='g', rotation=11)
print("Set text")

ax2.tick_params(axis='both', which='major', labelsize=16)
print("Set ticks")
fig2.show()
fig2.savefig(f"{time.time()}-dho.pdf")
