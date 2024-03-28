import numpy as np
import tensorflow as tf
from .MVP_Data_Creation_Josh import slimplecticSoln
import matplotlib.pyplot as plt

# Training Variables: Can be changed
EPOCHS = 2000
TRAINING_TIMESTEPS = 12
TRAINING_DATASIZE = 20480
dataName = "HarmonicOscillator"

XName = "Data/" + dataName + "/xData.npy"
YName = "Data/" + dataName + "/yData.npy"

# Data Variables: Do not change unless data is regenerated
DATASIZE = 20480
TIMESTEPS = 40


def custom_loss(y_True, y_Pred):
    ypnp = y_Pred.numpy()
    ytnp = y_True.numpy()
    print(ypnp[0])
    print(y_True.shape)
    totalLoss = 0

    for index in range(0, ypnp.shape[0]):
        qP = slimplecticSoln(TIMESTEPS, ypnp[index])[0]
        qT = slimplecticSoln(TIMESTEPS, ytnp[index])[0]

        # RMS of the true vs predicted q values
        totalLoss += np.sqrt(np.sum((qT - qP) ** 2))

    print(f"ALPHA Returning loss {totalLoss}")
    return totalLoss / (ypnp.shape[0])


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


X, Y, timestep_filter, datasize_filter = loadData(XName, YName)


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
    model.compile(optimizer='adam', loss=custom_loss, run_eagerly=True)

    # Train the model
    model_loss = model.fit(X, Y, epochs=EPOCHS, batch_size=batchsize, validation_split=0.2, verbose=2)

    # return loss
    loss_list = model_loss.history["loss"]
    val_loss_list = model_loss.history["val_loss"]

    return loss_list, val_loss_list


if __name__ == "__main__":
    loss_list, val_loss_list = runModel(1, [5, 5, 5, 5], [1, 1, 1, 1], 0.2, 128)

    # Make Plot of Loss
    title = dataName + ", " + "datapoints: " + str(timestep_filter) + ", Datasize: " + str(datasize_filter)
    epochs = [i for i in range(1, EPOCHS + 1)]
    plt.plot(epochs, loss_list)
    plt.plot(epochs, val_loss_list)
    plt.title(title)
    plt.show()
    print("min loss: " + str(min(loss_list)))
