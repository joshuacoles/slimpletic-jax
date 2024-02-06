import random
import numpy as np
import tensorflow as tf
from MVP_Data_Creation import slimplecticSoln

EPOCHS = 1000
DATASIZE = 64

def genData():
    q_data, pi_data, L_data = [], [], []

    for _ in range(DATASIZE):
        q, p, l = slimplecticSoln()
        q_data.append(q[0])
        pi_data.append(p[0])
        L_data.append(l)

    X = np.array([q_data, pi_data]).reshape((DATASIZE,1001,2))
    Y = np.array(L_data)
    return X, Y

# Model Definition
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=1000, input_shape=(1001, 2), return_sequences=True),

    # tf.keras.layers.LSTM(units=1000, input_shape=(1001, 2), return_sequences=True, kernel_regularizer=tf.keras.regularizers.L1L2()),
    # Optional Other Layers HERE
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=6)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
X, Y = genData()

# Data Normalization
mean_X, std_X = np.mean(X), np.std(X)
X_normalized = (X - mean_X) / std_X


# Train the model
model.fit(X_normalized, Y, epochs=EPOCHS, batch_size=64, validation_split=0.2)


# Data Normalization for Test Set
XTest, YTest = genData()
XTest_normalized = (XTest - mean_X) / std_X

# Evaluate the model
test_loss = model.evaluate(XTest_normalized, YTest)

print(test_loss)

