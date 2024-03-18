import jax
import jaxopt
import matplotlib.pyplot as plt
from jax import jit, grad, numpy as jnp
import numpy as np

from loss_fn.graph_helpers import plot_variation_graph, create_plots, plot_comparison
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle

dof = 1


def lagrangian_family(q, v, _, embedding):
    """
    A 3 parameter family of Lagrangians for 1D coordinates of the form

    \\begin{equation}
    L(q, v) = q^2 \\cdot e_0 + v^2 \\cdot e_1 + qv e_2
    \\end{equation}

    where $e \\in \\R^3$ is the embedding vector.
    """
    v = (embedding[0] * (q[0] ** 2) +
         embedding[1] * (v[0] ** 2) +
         embedding[2] * (q[0] * v[0]))

    return v


def make_solver():
    q0 = jnp.array([0.0])
    pi0 = jnp.array([1.0])
    t0 = 0
    iterations = 100
    dt = 0.1
    t = t0 + dt * np.arange(0, iterations + 1)

    ggl_bundle = GGLBundle(r=2)

    # The system which will be used when computing the loss function.
    embedded_system_solver = SolverScan(DiscretisedSystem(
        dt=dt,
        ggl_bundle=ggl_bundle,
        lagrangian=lagrangian_family,
        k_potential=None,
        pass_additional_data=True
    ))

    return t, lambda embedding: embedded_system_solver.integrate(
        q0=q0,
        pi0=pi0,
        t0=t0,
        iterations=iterations,
        additional_data=embedding
    )


t, solve = make_solver()


def rms(x, y):
    return jnp.sqrt(jnp.mean((x - y) ** 2))


@jit
def rms_both_loss_fn(embedding: jnp.ndarray, target_q: jnp.ndarray, target_pi: jnp.ndarray):
    """
    The most naive physically informed loss function for the embedding problem. It computes the RMS of the difference
    between the target and the actual q and pi values.

    We define "physically informed" to be a loss which is in some way related to the physics of the system.
    """
    q, pi = solve(embedding)
    return rms(q, target_q) + rms(pi, target_pi)


def embedding_rms_loss_fn(embedding: jnp.ndarray, true_embedding: jnp.ndarray):
    """
    I suppose an even more naive loss function for the embedding problem. It computes the RMS of the difference between
    the target and the actual embedding, this is not at all
    """
    return jnp.sqrt(jnp.mean((embedding - true_embedding) ** 2))


# T - V
# 1/2 * m * v^2 - 1/2 * k * q^2
true_embedding = jnp.array([-0.5, 0.5, 0])
target_q, target_pi = solve(true_embedding)

fig, variation_grid_spec, comparison_ax, loss_variation_size = create_plots(embedding_size=3, label="RMS (Both) Loss")

loss_fn = lambda trial: rms_both_loss_fn(trial, target_q, target_pi)

# embedding = jnp.array(np.random.rand(3))
embedding = jaxopt.GradientDescent(
    loss_fn,
    maxiter=1000,
    verbose=True,
).run(
    jnp.array(np.random.rand(3)),
).params

print(embedding)

plot_variation_graph(
    loss_fn,
    embedding,
    jnp.linspace(-1, 1, 100),
    fig,
    variation_grid_spec,
    loss_variation_size
)

comparison_ax.plot(t, solve(embedding)[0], label="Predicted", linestyle="--")
comparison_ax.plot(t, target_q, label="Expected")

fig.show()
