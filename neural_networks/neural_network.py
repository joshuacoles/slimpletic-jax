import os

os.environ["KERAS_BACKEND"] = "jax"

from neural_networks.physical_loss_layer import PhysicsLoss

import keras

from neural_networks.data.families import dho
from neural_networks.our_code_here import TRAINING_TIMESTEPS, \
    load_data_wrapped, create_layer

family = dho

inputs = keras.Input(shape=(TRAINING_TIMESTEPS + 1, 2))
lstm_1 = create_layer(50, True)(inputs)
lstm_2 = create_layer(40, True)(lstm_1)
lstm_3 = create_layer(20, True)(lstm_2)
flatten = keras.layers.Flatten()(lstm_3)
dense = keras.layers.Dense(units=dho.embedding_shape[0])(flatten)
physics_layer = PhysicsLoss(
    family=dho,
)([inputs, dense])

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
