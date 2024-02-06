import random

import numpy as np
import tensorflow as tf
from MVP_Data_Creation import slimplecticSoln

EPOCHS = 100
DATASIZE = 256


def genData():
    for i in range(0, DATASIZE):
        print(i)
        q, p, l = slimplecticSoln()
        q = q[0]
        p = p[0]
        if i == 0:
            q_data = [q]
            pi_data = [p]
            L_data = [l]
        else:
            q_data = np.concatenate((q_data, [q]))
            pi_data = np.concatenate((pi_data, [p]))
            L_data = np.concatenate((L_data, [l]))

    X = np.array([q_data, pi_data]).reshape(DATASIZE, 1001, 2)
    print(X.shape)
    Y = L_data
    print(Y.shape)
    return X, Y


# Model Definition
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=100, input_shape=(1001, 2), return_sequences=True, kernel_regularizer=tf.keras.regularizers.L1L2()),
    tf.keras.layers.LSTM(units=100, input_shape=(1001, 2), return_sequences=True,
                         kernel_regularizer=tf.keras.regularizers.L1L2()),
    # Optional Other Layers HERE
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=6)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # Adjust the loss based on your problem

X, Y = genData()
# Train the model
model.fit(X, Y, epochs=EPOCHS, batch_size=32, validation_split=0.2)

XTest, YTest = genData()
# Evaluate the model
test_loss = model.evaluate(XTest, YTest)

print(test_loss)
