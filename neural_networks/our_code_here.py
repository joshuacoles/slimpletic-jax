import keras
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from neural_networks.data.families import aengus_original
from neural_networks.data.generate_data_impl import setup_solver

# Training Variables: Can be changed
EPOCHS = 20
TRAINING_TIMESTEPS = 12
TRAINING_DATASIZE = 2

# Data Variables: Do not change unless data is regenerated
DATASIZE = 20480
TIMESTEPS = 40

# Solver
solve = setup_solver(
    family=aengus_original,
    iterations=TIMESTEPS
)


def create_layer(units, regularizer, i):
    if regularizer[i] == 1:
        return keras.layers.LSTM(units=units[i], input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                 return_sequences=True,
                                 kernel_regularizer=keras.regularizers.L1L2())
    else:
        return keras.layers.LSTM(units=units[i], input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                 return_sequences=True)


def create_model(layers: int, units: list, regulariser: list, dropout: float):
    """
    :param layers:
    :param units:
    :param regulariser:
    :param dropout:
    :return:
    """
    model = tf.keras.Sequential([
        *[create_layer(units, regulariser, i + 4 - layers) for i in range(layers)],
        keras.layers.Dropout(dropout),
        keras.layers.Flatten(),
        keras.layers.Dense(units=4)
    ])
    return model


def get_model() -> keras.Model:
    layers = 3
    units = [20, 15, 10, 5]
    regulariser = [1, 1, 1, 1]
    dropout = 0.25
    return create_model(layers, units, regulariser, dropout)


def get_data(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784)).astype("float32")
    x_test = np.reshape(x_test, (-1, 784)).astype("float32")
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset

def wrapped_solve(embedding: jnp.ndarray) -> jnp.ndarray:
    return solve(
        embedding,
        jnp.array([1.0]), jnp.array([1.0])
    )[0]
def loss_fn(y_true: jnp.ndarray, y_predicated: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.fori_loop(
        0, y_true.shape[0],
        lambda index, total_loss: total_loss + np.sqrt(
            np.sum((wrapped_solve(y_true[index]) - wrapped_solve(y_predicated)) ** 2)),
        0
    )

def get_loss_fn() -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """
    Returns a fn of the form:

        def loss_fn(y_true, y_pred):
            return loss
    """
    # Instantiate a loss function.
    return loss_fn
