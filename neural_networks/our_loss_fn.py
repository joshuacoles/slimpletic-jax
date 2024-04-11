import jax.lax
import numpy as np
import jax.numpy as jnp

from neural_networks.data.families import aengus_original
from neural_networks.data.generate_data_impl import setup_solver

TIMESTEPS = 40

solve = setup_solver(
    family=aengus_original,
    iterations=TIMESTEPS
)


def wrapped_solve(embedding: jnp.ndarray) -> jnp.ndarray:
    return solve(
        embedding,
        jnp.array([1.0]), jnp.array([1.0])
    )[0]


def loss_fn(y_true: np.ndarray, y_predicated: np.ndarray) -> np.ndarray:
    return jax.lax.fori_loop(
        0, y_true.shape[0],
        lambda index, total_loss: total_loss + np.sqrt(
            np.sum((wrapped_solve(y_true[index]) - wrapped_solve(y_predicated)) ** 2)),
        0
    )
