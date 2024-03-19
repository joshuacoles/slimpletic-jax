from functools import partial
from typing import Callable

from jax import numpy as jnp, jit


def rms(x, y):
    return jnp.sqrt(jnp.mean((x - y) ** 2))


@partial(jit, static_argnums=(0,))
def rms_both_loss_fn(solve, embedding: jnp.ndarray, target_q: jnp.ndarray, target_pi: jnp.ndarray):
    """
    The most naive physically informed loss function for the embedding problem. It computes the RMS of the difference
    between the target and the actual q and pi values.

    We define "physically informed" to be a loss which is in some way related to the physics of the system.
    """
    q, pi = solve(embedding)
    return rms(q, target_q) + rms(pi, target_pi)


def embedding_rms_loss_fn(embedding: jnp.ndarray, true_embedding: jnp.ndarray):
    """
    I suppose an even more naive loss function for the embedding problem. It computes the RMS of the difference between
    the target and the actual embedding, this is not at all
    """
    return jnp.sqrt(jnp.mean((embedding - true_embedding) ** 2))
