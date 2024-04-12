import keras
from typing import Callable
import jax
import jax.numpy as jnp
import tensorflow as tf
from neural_networks.data.families import aengus_original, dho
from neural_networks.data.generate_data_impl import setup_solver
from neural_networks.data import load_nn_data

dataName = "pure_normal-0"
family = dho

# Training Variables: Can be changed
EPOCHS = 20
TRAINING_TIMESTEPS = 12
TRAINING_DATASIZE = 2

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


def create_model(layers: int, units: list[int], regulariser: list[int], dropout: float):
    """
    :param layers:
    :param units:
    :param regulariser:
    :param dropout:
    :return:
    """
    model = keras.Sequential([
        *[create_layer(units[-i], regulariser[-i] == 1) for i in range(layers)],
        keras.layers.Dropout(dropout),
        keras.layers.Flatten(),
        keras.layers.Dense(units=family.embedding_shape[0])
    ])
    return model


def get_model() -> keras.Model:
    layers = 3
    units = [5, 5, 5, 3]
    regulariser = [1, 1, 1, 1]
    dropout = 0.25
    return create_model(layers, units, regulariser, dropout)


def get_data(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    # Reserve 10,000 samples for validation.
    validation_cutoff = 10000

    x_train, y_train = load_nn_data(family, dataName)
    x_train = x_train[:,:TRAINING_TIMESTEPS+1,:]

    x_val = x_train[-validation_cutoff:]
    y_val = y_train[-validation_cutoff:]
    x_train = x_train[:-validation_cutoff]
    y_train = y_train[:-validation_cutoff]

    # Prepare the training dataset.
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    # ValueError: Input 0 of layer "dense" is incompatible with the layer: expected axis -1 of input shape to have value 65, but received input with shape (32, 205)
    return train_dataset, val_dataset


def wrapped_solve(embedding: jnp.ndarray) -> jnp.ndarray:
    return solve(
        embedding,
        jnp.array([1.0]), jnp.array([1.0])
    )[0]


def loss_fn(y_true: jnp.ndarray, y_predicated: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.fori_loop(
        0, y_true.shape[0],
        lambda index, total_loss: total_loss + jnp.sqrt(
            jnp.sum((wrapped_solve(y_true[index]) - wrapped_solve(y_predicated)) ** 2)),
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
