import csv

import numpy as np
import tensorflow as tf
import pickle

# Training Variables: Can be changed
EPOCHS = 500
TRAINING_TIMESTEPS = 12
TRAINING_DATASIZE = 10240
dataName = "HarmonicOscillator"

XName = "Data/" + dataName + "/xData.npy"
YName = "Data/" + dataName + "/yData.npy"

# Data Variables: Do not change unless data is regenerated
DATASIZE = 20480
TIMESTEPS = 40

def loadData(XData, YData):
    X = np.load(XData)
    Y = np.load(YData)

    # Selecting first TRAINING_TIMESTEPS amount of time series data
    if TRAINING_TIMESTEPS < TIMESTEPS:
        timestep_filter = TRAINING_TIMESTEPS
    else:
        timestep_filter = TIMESTEPS
    if TRAINING_DATASIZE > 0 and TRAINING_DATASIZE < DATASIZE:
        datasize_filter = TRAINING_DATASIZE
    else:
        datasize_filter = DATASIZE

    X = X[:datasize_filter, :timestep_filter + 1, :]
    Y = Y[:datasize_filter, :]

    # Data Normalization
    mean_X, std_X = np.mean(X), np.std(X)
    X_normalized = (X - mean_X) / std_X

    print(X_normalized.shape, Y.shape)

    return (X_normalized, Y, timestep_filter, datasize_filter)


def create_layer(units, regularizer, i):
    if regularizer[i] == 1:
        return tf.keras.layers.LSTM(units=units[i], input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                    return_sequences=True,
                                    kernel_regularizer=tf.keras.regularizers.L1L2())
    else:
        return tf.keras.layers.LSTM(units=units[i], input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                    return_sequences=True)


def runModel(layers: int, units: list, regulariser: list, dropout: float, batchsize: int):
    """
    :param layers: int identifying the number of layers
    :param units: list indexing how many units each layer has
    :param regulariser: list indexing which layers have regularisers
    :param dropout: float in [0.05,0.1,0.2,0.3,0.4]
    :param batchsize: 64,128,256 are usual values
    :return: Loss and ValLoss Lists indexed by Epoch
    """

    # Model Definition
    model = tf.keras.Sequential([
        *[create_layer(units, regulariser, i + 4 - layers) for i in range(layers)],
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4)
    ])

    # Compile the model
    # 'mean_squared_error'
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_loss = model.fit(X, Y, epochs=EPOCHS, batch_size=batchsize, validation_split=0.2, verbose=2)

    # return loss
    loss_list = model_loss.history["loss"]
    val_loss_list = model_loss.history["val_loss"]

    return loss_list, val_loss_list



if __name__ == "__main__":
    # Creates Data
    X, Y, timestep_filter, datasize_filter = loadData(XName, YName)
    with open("HyperParams.pkl",'rb') as pkl:
        permutations = pickle.load(pkl)
    with open("lossOutput.csv",'w') as file:
        wr = csv.writer(file)
        for setup in permutations:
            #setup format: [layer, size, prob, reglist, unitlist]
            wr.writerow(setup)
            loss,valLoss = runModel(setup[0],setup[4],setup[3],setup[2],setup[1])
            wr.writerow(loss)
            wr.writerow(valLoss)