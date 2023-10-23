import numpy as np
from jax import jit
from jax import numpy as jnp

m = 1.0
k = 1.0
ll = 1e-4 * np.sqrt(m * k)  # ll is $\lambda$ in


@jit
def conservative_lagrangian_jax(q, v):
    """Conservative Lagrangian for a 1D harmonic oscillator."""
    return 0.5 * v ** 2 - 0.5 * q ** 2


@jit
def non_conservative_component_jax(q_p, v_m):
    """Non-conservative component of the Lagrangian for a 1D harmonic oscillator."""
    return -q_p * v_m


def generate_collocation_points(r, precision):
    """Generate collocation points and weights for Gauss-Lobatto quadrature."""
    return jnp.arange(0, 1, 1 / r), jnp.array([1 / r] * r), jnp.array([1] * r ** 2).reshape(r, r)

