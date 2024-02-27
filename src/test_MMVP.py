import numpy as np
import tensorflow as tf
from MVP_Data_Creation import slimplecticSoln
import matplotlib.pyplot as plt

# Training Variables: Can be changed
EPOCHS = 20000
TRAINING_TIMESTEPS = 10
TRAINING_DATASIZE = 256
XName = "xData_1.npy"
YName = "yData_1.npy"

# Data Variables: Do not change unless data is regenerated
DATASIZE = 20480
TIMESTEPS = 40

def loadData(XData,YData):

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
    Y = Y[:datasize_filter,:]

    # Data Normalization
    mean_X, std_X = np.mean(X), np.std(X)
    X_normalized = (X - mean_X) / std_X

    print(X_normalized.shape, Y.shape)

    return(X_normalized,Y, timestep_filter, datasize_filter)

X,Y,timestep_filter, datasize_filter = loadData(XName,YName)

if __name__ == "__main__":
    # Model Definition
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=5 * TRAINING_TIMESTEPS, input_shape=(TRAINING_TIMESTEPS + 1, 2),
                             return_sequences=True, kernel_regularizer=tf.keras.regularizers.L1L2()),
        # Optional Other Layers HERE
        # tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=5)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_loss = model.fit(X, Y, epochs=EPOCHS, batch_size=32, validation_split=0.2, verbose=0)

    # Make Plot of Loss
    title = XName[:-4] + ", " + "datapoints: " + str(timestep_filter) + ", Datasize: " + str(datasize_filter)
    loss_list = model_loss.history["loss"]
    epochs = [i for i in range(1, EPOCHS + 1)]
    plt.plot(epochs, loss_list)
    plt.title(title)
    plt.show()
    print("min loss: " + str(min(loss_list)))
