import keras
from typing import Callable
import numpy as np
import tensorflow as tf


def get_model() -> keras.Model:
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


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


def get_loss_fn() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a fn of the form:

        def loss_fn(y_true, y_pred):
            return loss
    """
    # Instantiate a loss function.
    return keras.losses.CategoricalCrossentropy(from_logits=True)
