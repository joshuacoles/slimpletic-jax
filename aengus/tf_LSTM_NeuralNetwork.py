import jax
import numpy as np
import tensorflow as tf
import jax.numpy as jnp

from neural_networks.data.families import aengus_original
from neural_networks.data.generate_data_impl import setup_solver
import matplotlib.pyplot as plt

strategy = tf.distribute.MirroredStrategy()

# Training Variables: Can be changed
EPOCHS = 20
TRAINING_TIMESTEPS = 12
TRAINING_DATASIZE = 2
dataName = "HarmonicOscillator"

XName = "Data/" + dataName + "/xData.npy"
YName = "Data/" + dataName + "/yData.npy"

# Data Variables: Do not change unless data is regenerated
DATASIZE = 20480
TIMESTEPS = 40

solve = setup_solver(
    family=aengus_original,
    iterations=TIMESTEPS
)

"""
def solve_wrapper(y_pred, y_true):
    def solve_numpy(y_pred_numpy, y_true_numpy):
        # Your solve function implementation here
        return solve(y_pred_numpy, y_true_numpy)

    solved = tf.py_function(solve_numpy, [y_pred, y_true], tf.float32)
    return solved
"""


def solveWrap(y_elem):
    solved = tf.py_function(
        solve,
        [y_elem, jnp.array([1.0]), jnp.array([1.0])],
        Tout=tf.float32
    )
    return solved


def custom_loss(y_True, y_Pred):
    totalLoss = 0

    for index in range(0, y_True.shape[0]):
        qP = solve(jnp.array(y_Pred[index], dtype=jnp.float32), jnp.array([1.0]), jnp.array([1.0]))[0]
        qT = solve(jnp.array(y_True[index], dtype=jnp.float32), jnp.array([1.0]), jnp.array([1.0]))[0]

        # RMS of the true vs predicted q values
        totalLoss += np.sqrt(np.sum((qT - qP) ** 2))

    print(f"ALPHA Returning loss {totalLoss}")
    return totalLoss / (y_True.shape[0])


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


def create_model(layers: int, units: list, regulariser: list, dropout: float):
    """
    :param layers:
    :param units:
    :param regulariser:
    :param dropout:
    :return:
    """
    model = tf.keras.Sequential([
        *[create_layer(units, regulariser, i + 4 - layers) for i in range(layers)],
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4)
    ])
    return model


def runModel(layers: int, units: list, regulariser: list, dropout: float, batchsize: int):
    """
    :param layers: int identifying the number of layers
    :param units: list indexing how many units each layer has
    :param regulariser: list indexing which layers have regularisers
    :param dropout: float in [0.05,0.1,0.2,0.3,0.4]
    :param batchsize: 64,128,256 are usual values
    :return: Loss and ValLoss Lists indexed by Epoch
    """
    with strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

    with strategy.scope():
        model = create_model(layers, units, regulariser, dropout)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def loss_and_grads(params, X, Y):
        predictions = model(X, training=True)
        loss = custom_loss(Y, predictions)
        return loss, jax.grad(lambda params: loss)(params)

    def train_step(X, Y):
        print(type(model.trainable_variables[0]))
        loss, gradients = jax.value_and_grad(loss_and_grads)(model.trainable_variables, X, Y)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(Y, predictions)
        return loss

    train_step(X, Y)

    # Compile the model
    # 'mean_squared_error'
    # model.compile(optimizer='adam', loss=custom_loss, run_eagerly=True)

    # Train the model
    # model_loss = model.fit(X, Y, epochs=EPOCHS, batch_size=batchsize, validation_split=0.2, verbose=2)

    # return loss
    # loss_list = model_loss.history["loss"]
    # val_loss_list = model_loss.history["val_loss"]

    # return loss_list, val_loss_list


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
