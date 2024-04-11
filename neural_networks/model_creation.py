import keras

TRAINING_TIMESTEPS = 12


def create_layer(units: list[int], regularizer: list[int], i: int):
    if regularizer[i] == 1:
        return keras.layers.LSTM(units=units[i], input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                 return_sequences=True,
                                 kernel_regularizer=keras.regularizers.L1L2())
    else:
        return keras.layers.LSTM(units=units[i], input_shape=(TRAINING_TIMESTEPS + 1, 2),
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
        *[create_layer(units, regulariser, i + 4 - layers) for i in range(layers)],
        keras.layers.Dropout(dropout),
        keras.layers.Flatten(),
        keras.layers.Dense(units=4)
    ])
    return model
