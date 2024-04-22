# %%
import jax
import jax.numpy as jnp
import jaxopt

import neural_networks.data.families as families
from neural_networks.data.generate_data_impl import setup_solver

solver = setup_solver(
    family=families.dho,
    iterations=50
)

q0 = jnp.array([0.0])
pi0 = jnp.array([1.0])

true_embedding = jnp.array([2.0, 3.0, 1.0])
true_q, true_pi = solver(
    true_embedding,
    q0,
    pi0
)


def converged(embedding):
    return jnp.linalg.norm(embedding - true_embedding) < 0.1


def loss_fn(embedding):
    q_weight = 1
    pi_weight = 1

    a, b = solver(
        embedding,
        q0,
        pi0
    )

    rms = jnp.sqrt(jnp.mean(q_weight * (a - true_q) ** 2 + pi_weight * (b - true_pi) ** 2))

    return rms


gradient_descent = jaxopt.GradientDescent(loss_fn, maxiter=500)


def test(initial_embedding):
    result = gradient_descent.run(initial_embedding)
    return converged(result.params)


rng = jax.random.PRNGKey(0)
initial_embedding = jax.random.uniform(rng, (1000, 3))

result = jax.vmap(jax.jit(test))(initial_embedding)


# %%

def convert(emb_4):
    return emb_4[1:] / emb_4[0]


def loss_fn_4(emb_4):
    return loss_fn(convert(emb_4))


def test_4(initial_embedding):
    result = gradient_descent_4.run(initial_embedding)
    return converged(convert(result.params))


gradient_descent_4 = jaxopt.GradientDescent(loss_fn_4, maxiter=200)

rng = jax.random.PRNGKey(0)
initial_embedding_4 = jax.random.uniform(rng, (500, 4))

result_2 = jax.vmap(jax.jit(test_4))(initial_embedding_4)
