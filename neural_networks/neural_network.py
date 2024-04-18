import os
import datetime

os.environ["KERAS_BACKEND"] = "jax"

from neural_networks.physical_loss_layer import PhysicsLoss

import keras
import pandas as pd
import jax

from neural_networks.data.families import dho
from neural_networks.our_code_here import create_layer
from neural_networks.data import project_data_root, load_nn_data

TRAINING_TIMESTEPS = 100
family = dho

inputs = keras.Input(shape=(TRAINING_TIMESTEPS + 1, 2))
lstm_1 = create_layer(50, True)(inputs)
lstm_2 = create_layer(40, True)(lstm_1)
lstm_3 = create_layer(20, True)(lstm_2)
dropout = keras.layers.Dropout(0.2)(lstm_3)
flatten = keras.layers.Flatten()(dropout)
dense = keras.layers.Dense(units=family.embedding_shape[0])(flatten)
physics_layer = PhysicsLoss(family=dho, )([inputs, dense])

model = keras.Model(
    inputs=inputs,
    outputs=physics_layer,
)

model.compile(
    optimizer="adam",
    metrics=["mae"]
)

x, y = load_nn_data(
    'dho',
    'josh',
    filter_bad_data=True,
    maximum_value=10 ** 5,
    timestep_cap=TRAINING_TIMESTEPS,
    datasize_cap=1_000_000
)
#
# jax.profiler.start_trace("/Users/joshuacoles/Developer/checkouts/fyp/slimplectic-jax/data/logs", create_perfetto_link=True)

history = model.fit(
    x,
    y,
    epochs=4,
    batch_size=512,
    validation_split=0.2,
)

hist_df = pd.DataFrame(history.history)

key = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_data_root = project_data_root.joinpath(key)
model_data_root.mkdir(parents=True, exist_ok=True)

# or save to csv:
hist_csv_file = 'history.csv'
with open(model_data_root / hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

model.save(model_data_root / "model.keras")

# jax.profiler.stop_trace()
