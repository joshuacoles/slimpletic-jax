import numpy as np
from slimpletic import DiscretisedSystem, GGLBundle, SolverManual
import jax
import jax.numpy as jnp

power_series_order = 2
rng = np.random.default_rng()


def lagrangian_family(q, v, _, embedding):
    return jax.lax.fori_loop(0, power_series_order,
                             lambda i, acc: acc +
                                            (embedding[2 * i] * q[0] ** (i + 1)) +
                                            (embedding[2 * i + 1] * v[0] ** (i + 1)),
                             0.0) + embedding[-1]


system = DiscretisedSystem(
    ggl_bundle=GGLBundle(r=0),
    dt=0.1,
    lagrangian=lagrangian_family,
    k_potential=None,
    pass_additional_data=True,
)

solver = SolverManual(system)


def generate_trajectory(time_steps: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Generates a trajectory of a simple harmonic oscillator with a random lagrangian within the embedding family.
    :param time_steps: The number of time steps to generate forwards, will result in time_steps + 1 data points.
    :return:
    """
    embedding = jnp.array(rng.uniform(-20, 20, power_series_order * 2 + 1))

    q_slim, pi_slim = solver.integrate(
        q0=jnp.array([1.0]),
        pi0=jnp.array([1.0]),
        t0=0,
        iterations=time_steps,
        additional_data=embedding,
        result_orientation='coordinate'
    )

    # adding noise:
    q_slim += np.random.normal(0, abs(np.mean(q_slim.flatten()) / 500), np.shape(q_slim))
    pi_slim += np.random.normal(0, abs(np.mean(pi_slim.flatten()) / 500), np.shape(pi_slim))
    return q_slim, pi_slim, embedding


def generate_training_data(data_size: int, time_steps: int):
    qs, pis, embeddings = [], [], []

    for _ in range(data_size):
        if _ % 100 == 0:
            print(_)

        q, p, lagrangian_embedding = generate_trajectory(time_steps)
        qs.append(q[0])
        pis.append(p[0])
        embeddings.append(lagrangian_embedding)

    x_data = np.array([qs, pis]).reshape((data_size, time_steps + 1, 2))
    y_data = np.array(embeddings)

    return x_data, y_data


data_size = 20480
time_steps = 40

if __name__ == "__main__":
    X, Y = generate_training_data(data_size, time_steps)

    np.save("xData_lowNoise", X)
    np.save("yData_lowNoise", Y)
