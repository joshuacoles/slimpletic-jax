import jax.numpy as jnp

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
