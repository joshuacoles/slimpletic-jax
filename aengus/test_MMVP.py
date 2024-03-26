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
    print(y_Pred.numpy)
    q_true, q_pred = [], []
    qP = slimplecticSoln(TIMESTEPS, False, 1, y_Pred)[0]
    qT = slimplecticSoln(TIMESTEPS, False, 1, y_True)[0]
    q_pred.append(qP[0])
    q_true.append(qT[0])

    loss = 0
    for i in enumerate(qT):
        loss += (qT[i] - qP[i]) ** 2

    return loss ** 0.5


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


def create_layer(units, regularizer):
    return tf.keras.layers.LSTM(units=50 * TRAINING_TIMESTEPS, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                                return_sequences=True,
                                kernel_regularizer=tf.keras.regularizers.L1L2())


if __name__ == "__main__":
    # Model Definition
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50 * TRAINING_TIMESTEPS, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                             return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.L1L2()),
        tf.keras.layers.LSTM(units=25 * TRAINING_TIMESTEPS, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                             return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.L1L2()),
        tf.keras.layers.LSTM(units=10 * TRAINING_TIMESTEPS, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                             return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.L1L2()),
        tf.keras.layers.LSTM(units=5 * TRAINING_TIMESTEPS, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                             return_sequences=True, kernel_regularizer=tf.keras.regularizers.L1L2()),
        # Optional Other Layers HERE
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4)
    ])

    # Compile the model
    # 'mean_squared_error'
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_loss = model.fit(X, Y, epochs=EPOCHS, batch_size=128, validation_split=0.2, verbose=2)

    # Make Plot of Loss
    title = dataName + ", " + "datapoints: " + str(timestep_filter) + ", Datasize: " + str(datasize_filter)
    loss_list = model_loss.history["loss"]
    val_loss_list = model_loss.history["val_loss"]
    epochs = [i for i in range(1, EPOCHS + 1)]
    plt.plot(epochs, loss_list)
    plt.plot(epochs, val_loss_list)
    plt.title(title)
    plt.show()
    print("min loss: " + str(min(loss_list)))
