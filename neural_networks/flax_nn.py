from typing import Sequence
import jax
import flax
import optax

from jax import random, numpy as jnp
from jax import config
from flax import linen as nn

config.update("jax_enable_x64", True)

dataName = "HarmonicOscillator"
XName = "Data/" + dataName + "/xData.npy"
YName = "Data/" + dataName + "/yData.npy"

# Data Variables: Do not change unless data is regenerated
DATASIZE = 20480
TIMESTEPS = 40

# Training Variables: Can be changed
EPOCHS = 2000
TRAINING_TIMESTEPS = 12
TRAINING_DATASIZE = 20480
dataName = "HarmonicOscillator"


def loadData(XData, YData):
    X = jnp.load(XData)
    Y = jnp.load(YData)

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
    mean_X, std_X = jnp.mean(X), jnp.std(X)
    X_normalized = (X - mean_X) / std_X

    print(X_normalized.shape, Y.shape)

    return (X_normalized, Y, timestep_filter, datasize_filter)


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


# Same as JAX version but using model.apply().
@jax.jit
def mse(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        predicted_embedding = model.apply(params, x)
        return jnp.dot(y - predicted_embedding, y - predicted_embedding) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


count = 100
timesteps = 50
dof = 1
data_rng, params_rng = random.split(random.key(0), 2)

x_normalized, y_data, TSteps, datasize = loadData(XName, YName)

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
