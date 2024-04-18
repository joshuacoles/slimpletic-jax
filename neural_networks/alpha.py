import os

# This guide can only be run with the JAX backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import numpy as np
import jax.numpy as jnp

from neural_networks.data import load_nn_data
from neural_networks.data.families import dho
from neural_networks.our_code_here import create_model_layers, rms, vmapped_solve, TRAINING_TIMESTEPS, load_data_wrapped

family = dho


class CustomModel(keras.Sequential):
    def __init__(self):
        super().__init__([
            keras.Input(shape=(TRAINING_TIMESTEPS + 1, 2)),

            keras.layers.LSTM(units=50, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                              return_sequences=True,
                              kernel_regularizer=keras.regularizers.L1L2()),

            keras.layers.LSTM(units=25, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                              return_sequences=True,
                              kernel_regularizer=keras.regularizers.L1L2()),

            keras.layers.LSTM(units=15, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                              return_sequences=True,
                              kernel_regularizer=keras.regularizers.L1L2()),

            keras.layers.LSTM(units=5, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                              return_sequences=True,
                              kernel_regularizer=keras.regularizers.L1L2()),

            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(family.embedding_shape[0])
        ])

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    def loss_fn(self, x, y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)

    def compute_loss_and_updates(
            self,
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            training=False,
    ):
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            training=training,
        )
        loss = self.loss_fn(x, y, y_pred)
        return loss, (y_pred, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y = data

        # Get the gradient function.
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        # Compute the gradients.
        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            training=True,
        )

        # Update trainable variables and optimizer variables.
        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        # Update metrics.
        loss_tracker_vars = metrics_variables[: len(self.loss_tracker.variables)]
        mae_metric_vars = metrics_variables[len(self.loss_tracker.variables):]

        loss_tracker_vars = self.loss_tracker.stateless_update_state(
            loss_tracker_vars, loss
        )
        mae_metric_vars = self.mae_metric.stateless_update_state(
            mae_metric_vars, y, y_pred
        )

        logs = {}
        logs[self.loss_tracker.name] = self.loss_tracker.stateless_result(
            loss_tracker_vars
        )
        logs[self.mae_metric.name] = self.mae_metric.stateless_result(mae_metric_vars)

        new_metrics_vars = loss_tracker_vars + mae_metric_vars

        # Return metric logs and updated state variables.
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.loss_tracker, self.mae_metric]


model = CustomModel()
model.compile(optimizer="adam", metrics=["mae"])

x, y = load_data_wrapped(
    'dho',
    'physical-accurate-0'
)

print(x.shape, y.shape)

# Just use `fit` as usual
model.fit(
    x,
    y,
    epochs=5
)
