import os
from functools import partial

# This guide can only be run with the JAX backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np
import jax.numpy as jnp

from neural_networks.data.families import dho
from neural_networks.our_code_here import rms, vmapped_solve, TRAINING_TIMESTEPS, \
    load_data_wrapped, get_model, create_layer

family = dho


@keras.saving.register_keras_serializable()
class PhysicsLoss(keras.layers.Layer):
    def loss_fn(self, x, y_pred):
        q_true, pi_true = x[:, :, 0], x[:, :, 1]
        q_predicted, pi_predicted = vmapped_solve(y_pred)
        q_predicted = q_predicted.reshape(q_true.shape)
        pi_predicted = pi_predicted.reshape(pi_true.shape)

        rms_q = rms(q_predicted, q_true)
        rms_pi = rms(pi_predicted, pi_true)
        jax.debug.print("rms_q {}", rms_q)
        jax.debug.print("rms_pi {}", rms_pi)

        physical_loss = rms_q + rms_pi
        non_negatives = jnp.mean(jax.lax.select(y_pred < 0, jnp.exp(-10 * y_pred), jnp.zeros_like(y_pred)))

        return physical_loss / 2 + jnp.clip(non_negatives, 0, 1000)

    # Defines the computation
    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        self.add_loss(self.loss_fn(x, y))

        return inputs[1]


inputs = keras.Input(shape=(TRAINING_TIMESTEPS + 1, 2))
lstm_1 = create_layer(50, True)(inputs)
lstm_2 = create_layer(40, True)(lstm_1)
lstm_3 = create_layer(20, True)(lstm_2)
flatten = keras.layers.Flatten()(lstm_3)
dense = keras.layers.Dense(units=dho.embedding_shape[0])(flatten)
physics_layer = PhysicsLoss()([inputs, dense])

model = keras.Model(
    inputs=inputs,
    outputs=physics_layer,
)

model.compile(
    optimizer="adam",
    metrics=["mae"]
)

x, y = load_data_wrapped(
    'dho',
    'josh',
    TRAINING_TIMESTEPS,
    datasize_cap=100_000
)

tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)

model.fit(
    x,
    y,
    epochs=2,
    batch_size=256,
    validation_split=0.2,
    callbacks=[tb_callback],
)

model.save("model.keras")
