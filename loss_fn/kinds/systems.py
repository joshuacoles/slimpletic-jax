from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from dataclasses import dataclass
from .families import Family, power_series_with_prefactor, lookup_family


@dataclass
class PhysicalSystem():
    key: str
    true_embedding: jnp.ndarray
    family: Family
    q0: jnp.ndarray
    pi0: jnp.ndarray

    def to_json(self):
        return {
            "key": self.key,
            "true_embedding": self.true_embedding.tolist(),
            "family": self.family.key,
            "q0": self.q0.tolist(),
            "pi0": self.pi0.tolist(),
        }

    @staticmethod
    def from_json(json):
        return PhysicalSystem(
            key=json["key"],
            true_embedding=jnp.array(json["true_embedding"]),
            family=lookup_family(json["family"]),
            q0=jnp.array(json["q0"]),
            pi0=jnp.array(json["pi0"]),
        )


shm_prefactor = PhysicalSystem(
    'shm_prefactor',
    true_embedding=jnp.array([-0.5, 0.5, 0, 1.0], dtype=jnp.float64),
    family=power_series_with_prefactor,
    q0=jnp.array([0.0], dtype=jnp.float64),
    pi0=jnp.array([1.0], dtype=jnp.float64),
)

systems = {
    'shm': shm_prefactor,
    'shm_prefactor': shm_prefactor,
}


def lookup_system(key: str) -> PhysicalSystem:
    return systems[key]
