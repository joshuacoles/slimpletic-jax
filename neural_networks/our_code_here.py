import keras
from typing import Callable
import numpy as np
import tensorflow as tf

import model_creation
import our_loss_fn


def get_model() -> keras.Model:
    model_creation.create_model(
        layers=1,
        units=[5, 5, 5, 5],
        regulariser=[1, 1, 1, 1],
        dropout=0.2,
    )

    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_data(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    # Reserve 10,000 samples for validation.
    validation_cutoff = 10000

    x_train = np.load("/Users/joshuacoles/Developer/checkouts/fyp/nn-take-2/Data/HarmonicOscillator/xData.npy")
    y_train = np.load("/Users/joshuacoles/Developer/checkouts/fyp/nn-take-2/Data/HarmonicOscillator/yData.npy")

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

    return train_dataset, val_dataset


def get_loss_fn() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return our_loss_fn.loss_fn
