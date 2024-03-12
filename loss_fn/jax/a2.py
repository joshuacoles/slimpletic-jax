import jax
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
import flax
from flax import linen as nn

class SimpleDense(nn.Module):
  features: int
  kernel_init: Callable = nn.initializers.lecun_normal()
  bias_init: Callable = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs):
    kernel = self.param('kernel',
                        self.kernel_init, # Initialization function
                        (inputs.shape[-1], self.features))  # shape info.
    y = jnp.dot(inputs, kernel)
    bias = self.param('bias', self.bias_init, (self.features,))
    y = y + bias
    return y

key1, key2 = random.split(random.key(0), 2)
x = random.uniform(key1, (4,4))

model = SimpleDense(features=3)
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameters:\n', params)
print('output:\n', y)

@jax.jit
def loss_fn(params, x_batched, y_batched):
    # Define the squared loss for a single pair (x,y)
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(y - pred, y - pred) / 2.0

    # Vectorize the previous to compute the average of the loss on all samples.
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


learning_rate = 0.3  # Gradient step size.
loss_grad_fn = jax.value_and_grad(loss_fn)

import optax

params = model.init(key2, x)  # Initialization call
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)

for i in range(1000):
    loss_val, grads = loss_grad_fn(params, x, y)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print('Loss step {}: '.format(i), loss_val)

