import datetime
import json
import os

import jaxopt
from jax import numpy as jnp
import numpy as np

from loss_fn.graph_helpers import plot_variation_graph, create_plots
import loss_fns as loss_fns
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle


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


def lagrangian_family_emb4(q, v, _, embedding):
    """
    A 4 parameter family of Lagrangians for 1D coordinates of the form

    \\begin{equation}
    L(q, v) = e_3 \\cdot (q^2 \\cdot e_0 + v^2 \\cdot e_1 + qv e_2)
    \\end{equation}

    where $e \\in \\R^3$ is the embedding vector.
    """
    v = embedding[3] * (embedding[0] * (q[0] ** 2) +
                        embedding[1] * (v[0] ** 2) +
                        embedding[2] * (q[0] * v[0]))

    return v


def make_solver(family):
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
        lagrangian=family,
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


# Choose family and target embedding
# t, solve = make_solver(lagrangian_family)
# true_embedding = jnp.array([-0.5, 0.5, 0])
# system_label = "SHM emb3"

t, solve = make_solver(lagrangian_family_emb4)
true_embedding = jnp.array([-0.5, 0.5, 0, 1.0])
system_label = "SHM emb4"

# Choose loss function
loss_fn_type = loss_fns.q_and_pi_with_embedding_norm
loss_fn = loss_fn_type(solve, true_embedding)
loss_fn_label = loss_fn_type.__name__

batch = datetime.datetime.now().isoformat()

root = f"figures/{loss_fn_label}/{system_label}/{batch}"
os.makedirs(root)

target_q, _target_pi = solve(true_embedding)
maxiter = 1000

for i in range(10):
    fig, variation_grid_spec, comparison_ax, loss_variation_size = create_plots(
        embedding_size=true_embedding.size,
        label=f"{system_label}, {loss_fn_label}"
    )

    random_initial_embedding = jnp.array(np.random.rand(true_embedding.size))
    gradient_descent_result = jaxopt.GradientDescent(
        loss_fn,
        maxiter=maxiter,
        verbose=True,
    ).run(
        random_initial_embedding,
    )

    embedding = gradient_descent_result.params

    print(embedding)

    plot_variation_graph(
        loss_fn,
        embedding,
        jnp.linspace(-1, 1, 100),
        fig,
        variation_grid_spec,
        loss_variation_size
    )

    actual_q = solve(embedding)[0]
    comparison_ax.plot(t, actual_q, label="Predicted", linestyle="--")
    comparison_ax.plot(t, target_q, label="Expected")

    fig.show()
    fig.savefig(f"{root}/{i}.png")
    json.dump({
        "initial_embedding": random_initial_embedding.tolist(),
        "found_embedding": embedding.tolist(),
        "true_embedding": true_embedding.tolist(),
        "loss": float(loss_fn(embedding)),
        "maxiter": maxiter,
        "opt_state": gradient_descent_result.state._asdict(),
    }, open(f"{root}/{i}.json", "w"), indent=2)
