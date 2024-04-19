from functools import partial
from typing import Union

import jax
import keras
from jax import numpy as jnp

from neural_networks.data import Family, lookup_family
from neural_networks.data.generate_data_impl import setup_solver

PHYSICAL_COMPONENT_LOSS_MAXIMUM = 10 ** 15


def rms(x1: jnp.ndarray, x2: jnp.ndarray, weightings: Union[jnp.ndarray, None] = None) -> jnp.ndarray:
    if weightings is not None:
        return jnp.sqrt(jnp.mean(jnp.inner(weightings, (x1 - x2) ** 2), axis=0))
    else:
        return jnp.sqrt(jnp.mean((x1 - x2) ** 2))


@keras.saving.register_keras_serializable()
class PhysicsLoss(keras.layers.Layer):
    def __init__(self, family: Union[Family, str], **kwargs):
        super().__init__(**kwargs)
        self.family = family if isinstance(family, Family) else lookup_family(family)

    def get_config(self):
        config = super().get_config()
        config.update({
            "family": self.family.key,
        })
        return config

    def build(self, input_shape):
        x_shape, y_shape = input_shape
        self.timesteps = x_shape[1]
        super().build(input_shape)

        solver = setup_solver(family=self.family, iterations=self.timesteps - 1)
        self.solver = jax.vmap(jax.jit(solver))

    @jax.profiler.annotate_function
    @partial(jax.jit, static_argnums=(0,))
    def compute_predicted_trajectory(self, x, y_pred):
        # Convert to float64 for prediction
        q0 = x[:, 0, 0].reshape((y_pred.shape[0], 1)).astype(jnp.float64)
        pi0 = x[:, 0, 1].reshape((y_pred.shape[0], 1)).astype(jnp.float64)

        q_predicted, pi_predicted = self.solver(
            y_pred,
            q0,
            pi0,
        )

        return q_predicted, pi_predicted

    @jax.profiler.annotate_function
    def loss_fn(self, x, y_pred):
        q_true, pi_true = x[:, :, 0], x[:, :, 1]
        q_predicted, pi_predicted = self.compute_predicted_trajectory(x, y_pred)

        return self.compute_loss(pi_predicted, pi_true, q_predicted, q_true, y_pred)

    @jax.profiler.annotate_function
    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, pi_predicted, pi_true, q_predicted, q_true, y_pred):
        q_predicted = q_predicted.reshape(q_true.shape)
        pi_predicted = pi_predicted.reshape(pi_true.shape)

        weightings = jnp.arange(self.timesteps, 0, -1)
        rms_q = jnp.clip(rms(q_predicted, q_true, weightings), 0, PHYSICAL_COMPONENT_LOSS_MAXIMUM)
        rms_pi = jnp.clip(rms(pi_predicted, pi_true, weightings), 0, PHYSICAL_COMPONENT_LOSS_MAXIMUM)

        physical_loss = (rms_q + rms_pi) / 2
        non_negatives = jnp.mean(jax.lax.select(y_pred < 0, jnp.exp(-10 * y_pred), jnp.zeros_like(y_pred)))

        return physical_loss + jnp.clip(non_negatives, 0, 1000)

    # Defines the computation
    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        self.add_loss(self.loss_fn(x, y))

        return inputs[1]
