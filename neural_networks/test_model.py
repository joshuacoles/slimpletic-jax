import os
import sys

# This guide can only be run with the JAX backend.
os.environ["KERAS_BACKEND"] = "jax"

from neural_networks.data import lookup_family
from neural_networks.data.generate_data_impl import setup_solver
import keras
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Get the path from the first argument
path = sys.argv[1]

# Get the three floats from the command line arguments
float1 = float(sys.argv[2])
float2 = float(sys.argv[3])
float3 = float(sys.argv[4])

model = keras.models.load_model(path)

# We can graph longer than the model is trained on
graph_timesteps = 200
model_timesteps = model.layers[-1].timesteps

solver = setup_solver(
    family=lookup_family('dho'),
    iterations=max(graph_timesteps, model_timesteps)
)

# Generate data
true_embedding = jnp.array([float1, float2, float3])
q, pi = solver(
    true_embedding,
    jnp.array([0.0]),
    jnp.array([1.0])
)

# Reshape to match model input
model_input = jnp.concatenate([
    q[:model_timesteps],
    pi[:model_timesteps]
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
