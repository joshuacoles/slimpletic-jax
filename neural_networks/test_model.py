import json
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras.models
import slimpletic

import jax.numpy as jnp
import matplotlib.pyplot as plt

from neural_networks.data import project_data_root
from neural_networks.data.families import dho, lookup_family
from neural_networks.data.generate_data_impl import setup_solver

model_path = project_data_root.joinpath("training_data/2024-04-15T12:54:50.119015/model_4.keras")
config = json.load(model_path.parent.joinpath("config.json").open("r"))
solver = setup_solver(
    family=lookup_family(config["family"]),
    iterations=config["training_timesteps"]
)

model = keras.models.load_model(model_path)

# Generate data
true_embedding = jnp.array([1.0, 1.0, 1.0])
q, pi = solver(
    true_embedding,
    jnp.array([1.0]),
    jnp.array([1.0])
)

# Reshape to match model input
model_input = jnp.concatenate([q, pi], axis=-1).reshape(
    (1, model.input_shape[1], model.input_shape[2])
)

# Predict
embedding = model.call(model_input)

# Solve predicted embedding
pred_q, pred_pi = solver(
    embedding[0],
    jnp.array([1.0]),
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
