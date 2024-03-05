import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from data_creation import x_path, y_path

# Training Variables: Can be changed
training_epochs = 4000
training_time_steps = 10
training_datasize = 20480

# Profile from batches 10 to 15
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs_old', profile_batch=(0, 20))


def load_data(x_data_path, y_data_path):
    x = np.load(x_data_path)
    y = np.load(y_data_path)

    # Truncate trajectories at the r sizes, if these are greater than the generated data then this is a noop
    x = x[:training_datasize, :training_time_steps + 1, :]
    y = y[:training_datasize, :]

    # Data Normalization
    x_mean, x_std = np.mean(x), np.std(x)
    x_normalised = (x - x_mean) / x_std
    print(x_normalised.shape, y.shape)

    return x_normalised, y


if __name__ == "__main__":
    x_data, y_data = load_data(x_path, y_path)
    datasize_filter, timestep_filter, _ = x_data.shape

    with tf.device('/CPU:0'):
        x_data = tf.constant(x_data)
        y_data = tf.constant(y_data)

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
        model_training_history = model.fit(x_data, y_data, epochs=training_epochs, batch_size=64, validation_split=0.2, verbose=2,
                                           callbacks=[tb_callback])

    # Make Plot of Loss
    model_loss = model_training_history.history["loss"]
    plt.plot(np.arange(1, training_epochs + 1), model_loss)
    plt.title(f"{Path(x_path).stem}, datapoints: {timestep_filter}, Datasize: {datasize_filter}")
    plt.show()

    print(f"min loss: {min(model_loss)}")
