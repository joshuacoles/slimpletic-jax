import os

# This guide can only be run with the JAX backend.
os.environ["KERAS_BACKEND"] = "jax"

from neural_networks.data import lookup_family
from neural_networks.data.generate_data_impl import setup_solver
from neural_networks.our_code_here import TRAINING_TIMESTEPS

import jax
import keras
import jax.numpy as jnp
import matplotlib.pyplot as plt

model = keras.models.load_model('/Users/joshuacoles/Developer/checkouts/fyp/slimplectic-jax/neural_networks/model.keras')

solver = setup_solver(
    family=lookup_family('dho'),
    iterations=TRAINING_TIMESTEPS
)

# Generate data
true_embedding = jnp.array([1.0, 2.0, 3.0])
q, pi = solver(
    true_embedding,
    jnp.array([0.0]),
    jnp.array([1.0])
)

# Reshape to match model input
model_input = jnp.concatenate([
    q[:TRAINING_TIMESTEPS + 1],
    pi[:TRAINING_TIMESTEPS + 1]
], axis=-1).reshape(
    (1, model.input_shape[1], model.input_shape[2])
)

# Predict
embedding = model.call(model_input)

# Solve predicted embedding
pred_q, pred_pi = solver(
    embedding[0],
    jnp.array([0.0]),
    jnp.array([1.0])
)

print(f"True embedding: {true_embedding}")
print(f"Predicted embedding: {embedding[0]}")

t = jnp.arange(0, len(q))

# Plot
plt.title("Q")
plt.plot(t, q, label="True")
plt.plot(t, pred_q, label="Predicted")

plt.legend()
plt.show()

plt.title("Pi")
plt.plot(t, pi, label="True")
plt.plot(t, pred_pi, label="Predicted")

plt.legend()
plt.show()
