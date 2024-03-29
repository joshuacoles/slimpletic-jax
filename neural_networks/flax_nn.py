from functools import partial
import jax
import flax
import optax

from jax import random, numpy as jnp
from jax import config
from flax import linen as nn

from neural_networks.data.families import power_series_with_prefactor
from neural_networks.data.generate_data_impl import setup_solver
from neural_networks.data.load_data import load_data

config.update("jax_enable_x64", True)
family = power_series_with_prefactor
iterations = 12
solver = setup_solver(family, iterations)


class LSTMModel(nn.Module):
    lstm_features: int
    hidden_features: int
    output_features: int

    def setup(self):
        lstm_layer = nn.scan(nn.OptimizedLSTMCell,
                             variable_broadcast="params",
                             split_rngs={"params": False},
                             in_axes=0,
                             out_axes=0,
                             reverse=False)

        self.lstm_layer = lstm_layer(features=self.lstm_features)
        self.lstm_layer2 = lstm_layer(features=self.lstm_features)
        self.dense1 = nn.Dense(self.hidden_features)
        self.dense2 = nn.Dense(self.output_features)

    @nn.remat
    def __call__(self, x_batch):
        print('x_batch shape:', x_batch.shape)
        x = x_batch

        # why do we do this here?
        # x_batch has shape (batch, timesteps + 1, len(q) + len(pi))
        carry, hidden = self.lstm_layer.initialize_carry(
            rng=jax.random.PRNGKey(0),
            input_shape=x_batch.shape[1:],
        )
        _, x = self.lstm_layer((carry, hidden), x)

        # why do we do this here?
        # x_batch has shape (batch, timesteps + 1, len(q) + len(pi))
        carry, hidden = self.lstm_layer2.initialize_carry(
            rng=jax.random.PRNGKey(0),
            input_shape=x_batch.shape[1:],
        )
        _, x = self.lstm_layer2((carry, hidden), x)

        print('A x shape:', x.shape)
        x = x.flatten()
        print('B x shape:', x.shape)

        x = self.dense1(x)
        print('C x shape:', x.shape)
        x = nn.relu(x)
        print('D x shape:', x.shape)

        x = self.dense2(x)
        print('E x shape:', x.shape)

        return x


q0 = jnp.array([0.0], dtype=jnp.float64)
pi0 = jnp.array([1.0], dtype=jnp.float64)
solver = partial(solver, q0=q0, pi0=pi0)

# Same as JAX version but using model.apply().
@jax.jit
def mse(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, true_embedding):
        predicted_embedding = model.apply(params, x)
        true_q, _ = solver(true_embedding)
        predicted_q, _ = solver(predicted_embedding)

        diff = true_q - predicted_q
        diff_scalars = jax.vmap(jnp.linalg.norm)(diff)

        jax.debug.print("diff_scalars: {}", diff_scalars)

        return jnp.inner(diff_scalars, diff_scalars) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


count = 100
timesteps = 50
dof = 1
data_rng, params_rng = random.split(random.key(0), 2)

x_normalized, y_data = load_data(family, 'pure_normal')

x_normalized = x_normalized.astype(jnp.float32)
y_data = y_data.astype(jnp.float32)
dummy_x = jnp.zeros_like(x_normalized[0])

model = LSTMModel(
    lstm_features=16,
    hidden_features=32,
    output_features=4,
)

params = model.init(params_rng, dummy_x)
y = model.apply(params, x_normalized[0])

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
print('output:\n', y)

learning_rate = 0.3  # Gradient step size.
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)

for i in range(10 ** 4):
    loss_val, grads = loss_grad_fn(params, x_normalized, y_data)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print('Loss step {}: '.format(i), loss_val)
