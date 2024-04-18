import keras

from keras import Layer

from neural_networks.data.families import dho

dataFiles = [
    "physical-accurate-small-data-0",
    "physical-accurate-small-data-1",
    "physical-accurate-small-data-2",
    "physical-accurate-small-data-3"
]

family = dho

# NOTE: This is no longer used in neural_network, set the value there instead
TRAINING_TIMESTEPS = 100


def create_layer(unit: int, regularizer: bool):
    if regularizer:
        return keras.layers.LSTM(units=unit, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                 return_sequences=True,
                                 kernel_regularizer=keras.regularizers.L1L2())
    else:
        return keras.layers.LSTM(units=unit, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                 return_sequences=True)


def create_model_layers(layers: int, units: list[int], regulariser: list[int], dropout: float,
                        timestep_cap: int = TRAINING_TIMESTEPS) -> list[Layer]:
    lstm_layers = [[create_layer(units[-i], regulariser[-i] == 1)] for i in range(layers)]
    lstm_layers = [item for sublist in lstm_layers for item in sublist]

    return [
        keras.Input(shape=(timestep_cap + 1, 2)),
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
