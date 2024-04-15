import jax.numpy as jnp
import jax.random

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

from neural_networks.data import nn_data_path, save_nn_data
from neural_networks.data.families import power_series_with_prefactor, aengus_original, dho
from neural_networks.data.generate_data_impl import setup_solver

timesteps = 40
count = 1_000_000

# Change this for different random seeds
rng_seed = 0
rng = jax.random.PRNGKey(rng_seed)

# The family is the kind of system we are generating
family = dho
solver = setup_solver(
    family=family,
    iterations=timesteps,
)

# Choose a name for the population you are generating
population_name = f"physical-accurate-{rng_seed}"


def generate_population():
    """
    Generate $count number of embeddings which will be used as training data. These align with the format of the family
    specified above.
    """
    masses = jax.random.uniform(rng, (count, ), minval=0.1, maxval=10.0)
    spring_constants = jax.random.uniform(rng, (count, ), minval=0.1, maxval=10.0)
    damping_constants = jax.random.uniform(rng, (count, ), minval=0.1, maxval=10.0)

    return jnp.stack([masses, spring_constants, damping_constants], axis=-1)


lagrangian_embeddings = generate_population()

# We repeat the same initial conditions for each trajectory
q0 = jnp.repeat(jnp.array([0.0], dtype=jnp.float64)[jnp.newaxis, :], axis=0, repeats=count)
pi0 = jnp.repeat(jnp.array([1.0], dtype=jnp.float64)[jnp.newaxis, :], axis=0, repeats=count)

# Generate trajectories for each embedding
trajectories = jax.vmap(solver)(lagrangian_embeddings, q0, pi0)
trajectory_data = jnp.concatenate([trajectories[0], trajectories[1]], axis=-1)

# Write the data to a consistent location
save_nn_data(
    family,
    population_name,
    trajectory_data,
    lagrangian_embeddings
)
