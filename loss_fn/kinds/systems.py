from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from dataclasses import dataclass
from .families import Family, power_series_with_prefactor


@dataclass
class PhysicalSystem():
    key: str
    true_embedding: jnp.ndarray
    family: Family
    q0: jnp.ndarray
    pi0: jnp.ndarray


shm_prefactor = PhysicalSystem(
    'shm_prefactor',
    true_embedding=jnp.array([-0.5, 0.5, 0, 1.0], dtype=jnp.float64),
    family=power_series_with_prefactor,
    q0=jnp.array([0.0], dtype=jnp.float64),
    pi0=jnp.array([1.0], dtype=jnp.float64),
)
