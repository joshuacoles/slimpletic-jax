from typing import Callable, Any, Union
from jax import numpy as jnp


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

    return loss_fn


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

    return loss_fn


def q_only(solve: Callable, true_embedding: jnp.ndarray):
    """
    - Only care about q
    - Weight divergence as exp
    """
    target_q, target_pi = solve(true_embedding)

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        return rms(q, target_q)

    return loss_fn


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

    return loss_fn


def q_rms_embedding_norm_huber(solve: Callable, true_embedding: jnp.ndarray):
    target_q, target_pi = solve(true_embedding)
    delta = 0.1

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        embedding_norm = jnp.linalg.norm(embedding)

        # Calculate the absolute differences between vector elements
        abs_diffs = jnp.abs(jnp.diff(jnp.sort(embedding)))

        # Apply the Huber loss to the absolute differences
        huber_losses = jnp.where(
            abs_diffs <= delta,
            0.5 * abs_diffs ** 2,
            delta * abs_diffs - 0.5 * delta ** 2
        )

        # Sum and normalise the Huber losses
        huber_loss = jnp.sum(huber_losses) / embedding.size

        return rms(q, target_q) + embedding_norm + huber_loss

    return loss_fn


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

    return loss_fn


def q_argwhere_tol005(solve: Callable, true_embedding: jnp.ndarray):
    target_q, target_pi = solve(true_embedding)

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        delta_q = jnp.abs(target_q - q)
        argwhere = jnp.argwhere(
            delta_q > 0.05,
            size=1,
            fill_value=q.size
        )[0][0]
        return argwhere

    return loss_fn


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

    return loss_fn


def make_embedding_rms(_solve: Callable, true_embedding: jnp.ndarray):
    """
    I suppose an even more naive loss function for the embedding problem. It computes the RMS of the difference between
    the target and the actual embedding, this is not at all
    """

    def loss_fn(embedding: jnp.ndarray):
        return rms(embedding, true_embedding)

    return loss_fn


def q_rms_huber_embedding_norm(solve: Callable, true_embedding: jnp.ndarray):
    target_q, target_pi = solve(true_embedding)
    delta = 0.1

    def loss_fn(embedding: jnp.ndarray):
        q, _pi = solve(embedding)
        embedding_norm = jnp.linalg.norm(embedding)

        # Calculate the absolute differences between vector elements
        abs_diffs = jnp.abs(jnp.diff(jnp.sort(q - target_q)))

        # Apply the Huber loss to the absolute differences
        huber_losses = jnp.where(
            abs_diffs <= delta,
            0.5 * abs_diffs ** 2,
            delta * abs_diffs - 0.5 * delta ** 2
        )

        # Sum and normalise the Huber losses
        embedding_huber_loss = jnp.sum(huber_losses) / embedding.size

        return rms(q, target_q) + embedding_norm + embedding_huber_loss

    return loss_fn


loss_fns = {
    'rms_both': make_rms_both,
    'q_only_linear_weighted': q_only_linear_weighted,
    'embedding_rms': make_embedding_rms,
    'q_and_pi_with_embedding_norm': q_and_pi_with_embedding_norm,
    'q_only_with_embedding_norm_and_reverse_linear_weights': q_only_with_embedding_norm_and_reverse_linear_weights,
    'q_rms_embedding_norm_huber': q_rms_embedding_norm_huber,
    'q_only_with_embedding_norm': q_only_with_embedding_norm,
    'q_only': q_only,
    'q_rms_huber_embedding_norm': q_rms_huber_embedding_norm,
}

SimpleLossFn = Callable[[Callable, jnp.ndarray], jnp.ndarray]

LossFnWithConfig = Callable[[Callable, jnp.ndarray, Any], jnp.ndarray]


def lookup_loss_fn(key: str) -> Union[SimpleLossFn, LossFnWithConfig]:
    return loss_fns[key]
