from functools import partial
from typing import Callable

from jax import numpy as jnp, jit


def rms(x, y):
    return jnp.sqrt(jnp.mean((x - y) ** 2))


def make_rms_both(solve: Callable, true_embedding: jnp.ndarray):
    """
    The most naive physically informed loss function for the embedding problem. It computes the RMS of the difference
    between the target and the actual q and pi values.

    We define "physically informed" to be a loss which is in some way related to the physics of the system.
    """
    target_q, target_pi = solve(true_embedding)

    def loss_fn(embedding: jnp.ndarray):
        q, pi = solve(embedding)
        return rms(q, target_q) + rms(pi, target_pi)

    return jit(loss_fn)


def q_only_linear_weighted(solve: Callable, true_embedding: jnp.ndarray):
    """
    - Only care about q
    - Weight divergence as exp
    """
    target_q, target_pi = solve(true_embedding)
    weightings = jnp.arange(0, target_q.shape[0])

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        return jnp.sum(jnp.dot(weightings, jnp.abs(target_q - q)))

    return jit(loss_fn)


def q_only(solve: Callable, true_embedding: jnp.ndarray):
    """
    - Only care about q
    - Weight divergence as exp
    """
    target_q, target_pi = solve(true_embedding)

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        return rms(q, target_q)

    return jit(loss_fn)


def q_only_with_embedding_norm(solve: Callable, true_embedding: jnp.ndarray):
    """
    - Only care about q
    - Weight divergence as exp
    """
    target_q, target_pi = solve(true_embedding)

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        # We add the norm of the embedding to the loss function to stop the embedding from growing too large
        return rms(q, target_q) + jnp.linalg.norm(embedding)

    return jit(loss_fn)


def q_only_with_embedding_norm_and_reverse_linear_weights(solve: Callable, true_embedding: jnp.ndarray):
    """
    - Only care about q
    - Weight divergence as exp
    """
    target_q, target_pi = solve(true_embedding)
    weights = jnp.arange(target_q.shape[0], 0, -1) / target_q.shape[0]

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        # We add the norm of the embedding to the loss function to stop the embedding from growing too large
        return jnp.linalg.norm(embedding) + jnp.sum(jnp.dot(weights, jnp.abs(target_q - q)))

    return jit(loss_fn)


def q_and_pi_with_embedding_norm(solve: Callable, true_embedding: jnp.ndarray):
    """
    - Only care about q
    - Weight divergence as exp
    """
    target_q, target_pi = solve(true_embedding)

    def loss_fn(embedding: jnp.ndarray):
        q, pi = solve(embedding)
        # We add the norm of the embedding to the loss function to stop the embedding from growing too large
        return rms(q, target_q) + rms(pi, target_pi) + jnp.linalg.norm(embedding)

    return jit(loss_fn)


def make_embedding_rms(_solve: Callable, true_embedding: jnp.ndarray):
    """
    I suppose an even more naive loss function for the embedding problem. It computes the RMS of the difference between
    the target and the actual embedding, this is not at all
    """

    def loss_fn(embedding: jnp.ndarray):
        return rms(embedding, true_embedding)

    return jit(loss_fn)
