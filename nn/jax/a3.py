import flax.linen as nn
import jax, jax.numpy as jnp

x = jax.random.normal(jax.random.key(0), (2, 3))
layer = nn.LSTMCell(features=4)
carry = layer.initialize_carry(jax.random.key(1), x.shape)
variables = layer.init(jax.random.key(2), carry, x)
new_carry, out = layer.apply(variables, carry, x)

print(out)
