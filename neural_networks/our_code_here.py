import keras
import jax
import jax.numpy as jnp
import tensorflow as tf
import sys

from keras import Layer

from neural_networks.data.families import dho
from neural_networks.data.generate_data_impl import setup_solver
from neural_networks.data import load_nn_data

dataFiles = ["physical-accurate-small-data-0","physical-accurate-small-data-1","physical-accurate-small-data-2","physical-accurate-small-data-3"]
family = dho

# Training Variables: Can be changed
EPOCHS = 2
TRAINING_TIMESTEPS = 10
BATCH_SIZE = 128
SHUFFLE_SEED = None

# Solver
solve = setup_solver(
    family=family,
    iterations=TRAINING_TIMESTEPS
)


def create_layer(unit: int, regularizer: bool):
    if regularizer:
        return keras.layers.LSTM(units=unit, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                 return_sequences=True,
                                 kernel_regularizer=keras.regularizers.L1L2())
    else:
        return keras.layers.LSTM(units=unit, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                 return_sequences=True)


def create_model_layers(layers: int, units: list[int], regulariser: list[int], dropout: float) -> list[Layer]:
    lstm_layers = [[create_layer(units[-i], regulariser[-i] == 1)] for i in range(layers)]
    lstm_layers = [item for sublist in lstm_layers for item in sublist]

    return [
        keras.Input(shape=(TRAINING_TIMESTEPS + 1, 2)),
        *lstm_layers,
        keras.layers.Dropout(dropout),
        keras.layers.Flatten(),
        keras.layers.Dense(units=family.embedding_shape[0])
    ]


def create_model(layers: int, units: list[int], regulariser: list[int], dropout: float):
    return keras.Sequential(create_model_layers(layers, units, regulariser, dropout))


def get_model() -> keras.Model:
    layers = 5
    units = [50, 40, 30, 20, 10]
    regulariser = [1, 1, 1, 1, 1]
    dropout = 0.10
    return create_model(layers, units, regulariser, dropout)


def get_data(batch_size: int, dataName: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    # Reserve 10,000 samples for validation.
    validation_cutoff = 10_000
    x, y = load_data_wrapped(family, dataName)

    # Split into train and validation
    x_val = x[-validation_cutoff:]
    y_val = y[-validation_cutoff:]
    x_train = x[:-validation_cutoff]
    y_train = y[:-validation_cutoff]

    if not (jnp.all(jnp.isfinite(x_train)) and jnp.all(jnp.isfinite(y_train))):
        sys.exit('infs/NaNs in training data')
    if not (jnp.all(jnp.isfinite(x_val)) and jnp.all(jnp.isfinite(y_val))):
        sys.exit('infs/NaNs in validation data')

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(
        buffer_size=1024,
        seed=SHUFFLE_SEED,
    ).batch(batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset


def load_data_wrapped(family, dataName):
    maximum_value = 10 ** 5
    # Load data
    x, y = load_nn_data(family, dataName)
    row_indices = jnp.where(jnp.any(x > maximum_value, axis=1))[0]
    x_mask = jnp.ones(x.shape[0], dtype=bool).at[row_indices].set(False)
    y_mask = jnp.ones(y.shape[0], dtype=bool).at[row_indices].set(False)
    x = x[x_mask]
    y = y[y_mask]
    x = x[:, :TRAINING_TIMESTEPS + 1, :]
    return x, y


def wrapped_solve(embedding: jnp.ndarray) -> jnp.ndarray:
    return solve(
        embedding,
        jnp.array([0.0]), jnp.array([1.0])
    )


vmapped_solve = jax.vmap(fun=wrapped_solve, in_axes=(0,), )


def rms(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean((x1 - x2) ** 2))


def loss_fn(true_trajectory: jnp.ndarray, predicted_embedding: jnp.ndarray, true_embedding: jnp.ndarray) -> jnp.ndarray:
    q_true, pi_true = true_trajectory[:, :, 0], true_trajectory[:, :, 1]
    q_predicted, pi_predicted = vmapped_solve(predicted_embedding)
    q_predicted = q_predicted.reshape(q_true.shape)
    pi_predicted = pi_predicted.reshape(pi_true.shape)

    physical_loss = rms(q_predicted, q_true) + rms(pi_predicted, pi_true)

    return physical_loss / 2
