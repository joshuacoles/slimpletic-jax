import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_creation import data_size as generation_data_size, time_steps as generation_time_steps

# Training Variables: Can be changed
training_epochs = 4000
training_time_steps = 10
training_datasize = 20480
XName = "xData_lowNoise.npy"
YName = "yData_lowNoise.npy"


def load_data(x_data_path, y_data_path):
    x = np.load(x_data_path)
    y = np.load(y_data_path)

    # Truncate trajectories at this point, can't be greater than the length of the data
    time_steps_filter = np.min(training_time_steps, generation_time_steps)
    datasize_filter = np.min(training_datasize, generation_data_size)

    x = x[:datasize_filter, :time_steps_filter + 1, :]
    y = y[:datasize_filter, :]

    # Data Normalization
    x_mean, x_std = np.mean(x), np.std(x)
    x_normalised = (x - x_mean) / x_std
    print(x_normalised.shape, y.shape)

    return x_normalised, y, timestep_filter, datasize_filter


if __name__ == "__main__":
    X, Y, timestep_filter, datasize_filter = load_data(XName, YName)

    # Model Definition
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            units=5 * training_time_steps,
            input_shape=(training_time_steps + 1, 2),
            return_sequences=True
        ),

        # Optional Other Layers HERE
        # , kernel_regularizer=tf.keras.regularizers.L1L2()

        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=5)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model_loss = model.fit(X, Y, epochs=training_epochs, batch_size=64, validation_split=0.2, verbose=2)

    # Make Plot of Loss
    title = XName[:-4] + ", " + "datapoints: " + str(timestep_filter) + ", Datasize: " + str(datasize_filter)
    loss_list = model_loss.history["loss"]
    epochs = [i for i in range(1, training_epochs + 1)]
    plt.plot(epochs, loss_list)
    plt.title(title)
    plt.show()
    print("min loss: " + str(min(loss_list)))
