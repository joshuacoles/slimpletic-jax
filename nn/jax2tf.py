# import numpy as np
import tensorflow as tf
from data_creation import x_path, y_path, solver, q0, pi0, time_steps, power_series_order, rng
import jax.numpy as jnp
from jax.experimental import jax2tf

with tf.device('cpu'):
    def solver_fn(jax_embedding):
        return solver.integrate(
            q0=q0,
            pi0=pi0,
            t0=0,
            iterations=time_steps,
            additional_data=jax_embedding,
        )


    print("Converting solver_fn to tensorflow")
    converted = jax2tf.convert(solver_fn)
    print("Converting done")
    embedding = tf.constant(jnp.array(rng.uniform(-20, 20, power_series_order * 2 + 1)))
    print(f"Embedding created {embedding}")
    result = converted(embedding)
    print(f"Result: {result}")