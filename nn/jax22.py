import jax
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
import flax
from flax import linen as nn

# We create one dense layer instance (taking 'features' parameter as input)
model = nn.Dense(features=5)

key1, key2 = random.split(random.key(0))
x = random.normal(key1, (10,))  # Dummy input data
params = model.init(key2, x)  # Initialization call
jax.tree_util.tree_map(lambda x: x.shape, params)  # Checking output shapes

print(model.apply(params, x))

# Set problem dimensions.
n_samples = 20
x_dim = 10
y_dim = 5

# Generate random ground truth W and b.
key = random.key(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))
# Store the parameters in a FrozenDict pytree.
true_params = flax.core.freeze({'params': {'bias': b, 'kernel': W}})

# Generate samples with additional noise.
key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = jnp.dot(x_samples, W) + b + 0.1 * random.normal(key_noise, (n_samples, y_dim))
print('x shape:', x_samples.shape, '; y shape:', y_samples.shape)


# Same as JAX version but using model.apply().
@jax.jit
def loss_fn(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W,b: ', loss_fn(true_params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(loss_fn)

import optax

params = model.init(key2, x)  # Initialization call
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)

for i in range(1000):
    loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print('Loss step {}: '.format(i), loss_val)


# Take 2

class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)

            # Perform ReLU on all but the last layer
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x


x = random.uniform(key1, (4, 4))

model = ExplicitMLP(features=[3, 4, 5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(params)))
print('output:\n', y)
