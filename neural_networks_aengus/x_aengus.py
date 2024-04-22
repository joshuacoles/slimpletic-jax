import os

from neural_networks_aengus.data.generate_data_impl import setup_solver

# This guide can only be run with the JAX backend.
os.environ["KERAS_BACKEND"] = "jax"

import jax
import keras
import jax.numpy as jnp
import matplotlib.pyplot as plt

from neural_networks_aengus.data.families import dho, lookup_family
from neural_networks_aengus.our_code_here import TRAINING_TIMESTEPS
from neural_networks_aengus.beta import rms, vmapped_solve

family = dho


@keras.saving.register_keras_serializable()
class PhysicsLoss(keras.layers.Layer):
    """
    A massive hack of adding
    """
    def loss_fn(self, x, y_pred):
        q_true, pi_true = x[:, :, 0], x[:, :, 1]
        q_predicted, pi_predicted = vmapped_solve(y_pred)
        q_predicted = q_predicted.reshape(q_true.shape)
        pi_predicted = pi_predicted.reshape(pi_true.shape)

        physical_loss = rms(q_predicted, q_true) + rms(pi_predicted, pi_true)
        non_negatives = jnp.mean(jax.lax.select(y_pred < 0, jnp.exp(-10 * y_pred), jnp.zeros_like(y_pred)))

        return physical_loss / 2 + non_negatives

    # Defines the computation
    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        self.add_loss(self.loss_fn(x, y))

        return inputs[1]

model = keras.models.load_model('/Users/aengus/PycharmProjects/ai-physicist/neural_networks_aengus/ckpt/model_scaled_2.1.2/checkpoint.model.keras', custom_objects={
    'PhysicsLoss': PhysicsLoss
})

solver = setup_solver(
    family=lookup_family('dho'),
    iterations=TRAINING_TIMESTEPS
)

# Generate data
true_embedding = jnp.array([2.0,4.0,1.0])
q, pi = solver(
    true_embedding,
    jnp.array([0.0]),
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
    jnp.array([0.0]),
    jnp.array([1.0])
)

print(f"True embedding: {true_embedding}")
print(f"Predicted embedding: {embedding[0]}")

t = jnp.arange(0, len(q))

# Plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(t, q)
ax1.plot(t, pred_q)
ax1.set_title("Q")         # Title for the first subplot
ax1.set_xlabel("t")                    # X-axis label for the first subplot
ax1.set_ylabel("q")                    # Y-axis label for the first subplot

ax2.plot(t, pi, label="True")
ax2.plot(t, pred_pi, label="Predicted")
ax2.set_title(r'$\pi$')       # Title for the second subplot
ax2.set_xlabel("t")                    # X-axis label for the second subplot
ax2.set_ylabel(r'$\pi')                # Y-axis label for the second subplot
fig.legend()

plt.tight_layout()
plt.savefig("modelComp.jpg",format='jpg')
plt.show()
