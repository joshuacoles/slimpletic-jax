import keras
import jax
import jax.numpy as jnp
import tensorflow as tf
from neural_networks.data.families import dho
from neural_networks.data.generate_data_impl import setup_solver
from neural_networks.data import load_nn_data

dataName = "physical-accurate-0"
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
    # lstm_layers = [[create_layer(units[-i], regulariser[-i] == 1), keras.layers.LayerNormalization()] for i in range(layers)]
    lstm_layers = [[create_layer(units[-i], regulariser[-i] == 1)] for i in range(layers)]
    lstm_layers = [item for sublist in lstm_layers for item in sublist]

    model = keras.Sequential([
        keras.Input(shape=(TRAINING_TIMESTEPS + 1, 2)),
        *lstm_layers,
        keras.layers.Dropout(dropout),
        keras.layers.Flatten(),
        keras.layers.Dense(units=family.embedding_shape[0])
    ])
    return model


def get_model() -> keras.Model:
    layers = 4
    units = [32, 16, 8, 4]
    regulariser = [1, 1, 1, 1]
    dropout = 0.25
    return create_model(layers, units, regulariser, dropout)


def get_data(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    # Reserve 10,000 samples for validation.
    validation_cutoff = 10_000
    maximum_value = 10 ** 5

    # Load data
    x, y = load_nn_data(family, dataName)
    row_indices = jnp.where(jnp.any(x > maximum_value, axis=1))[0]
    x_mask = jnp.ones(x.shape[0], dtype=bool).at[row_indices].set(False)
    y_mask = jnp.ones(y.shape[0], dtype=bool).at[row_indices].set(False)

    x = x[x_mask]
    y = y[y_mask]

    x = x[:, :TRAINING_TIMESTEPS + 1, :]

    # Split into train and validation
    x_val = x[-validation_cutoff:]
    y_val = y[-validation_cutoff:]
    x_train = x[:-validation_cutoff]
    y_train = y[:-validation_cutoff]

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
        jnp.array([0.0]), jnp.array([1.0])
    )[0]


vmapped_solve = jax.vmap(fun=wrapped_solve, in_axes=(0,), )


def loss_fn(true_trajectory: jnp.ndarray, y_predicted: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    # return jnp.sqrt(jnp.sum(y_true - y_predicted) ** 2)

    q_predicted = vmapped_solve(y_predicted)
    q_true = true_trajectory[:, :, 0]

    # jax.debug.print("q_true: {}", q_true)
    # jax.debug.print("q_predicted: {}", q_predicted)

    residuals = (jnp.sum((q_true - q_predicted.reshape(q_true.shape)) ** 2))
    # jax.debug.print("Residuals: {}", residuals)
    # jax.debug.print("y_predicted: {}", y_predicted)
    return jnp.sqrt(residuals)
