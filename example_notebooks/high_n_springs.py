import time

import jax
import jax.numpy as jnp

from slimpletic import GGLBundle, DiscretisedSystem, SolverScan, SolverBatchedScan

rng = jax.random.key(0)

spring_count = 600#10_000
masses = jnp.absolute(jax.random.uniform(rng, (spring_count,)) * 1000)
spring_constants = jnp.absolute(jax.random.uniform(rng, (spring_count,)) * 1000)


def lagrangian(q, q_dot, t):
    kinetic_energy = 0.5 * jnp.sum(masses * q_dot ** 2)
    potential_energy = 0.5 * jnp.sum(spring_constants * q ** 2)
    return kinetic_energy - potential_energy


# Random initial conditions
q0 = jax.random.uniform(rng, (spring_count,))
pi0 = jax.random.uniform(rng, (spring_count,))

system = DiscretisedSystem(
    lagrangian=lagrangian,
    k_potential=None,
    dt=0.01,
    ggl_bundle=GGLBundle(r=0),
)

solver = SolverScan(system)
# solver = SolverBatchedScan(system, batch_size=100)

print("Integrating...")
start = time.time()
results = solver.integrate(
    q0=q0,
    pi0=pi0,
    iterations=1000,
    t0=0,
    result_orientation='coordinate',
)
print(f"Integration took {time.time() - start:.2f} seconds")
