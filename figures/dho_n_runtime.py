import time
import datetime

from neural_networks.data import project_data_root

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import pandas

from neural_networks.data.families import Family, dho
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle

r = 2
dt = 0.1

ggl_bundle = GGLBundle(r=2)
# The system which will be used when computing the loss function.
solver = SolverScan(DiscretisedSystem(
    dt=dt,
    ggl_bundle=ggl_bundle,
    lagrangian=dho.lagrangian,
    k_potential=dho.k_potential,
    pass_additional_data=True
))

# Run the system once to warm up the JIT
embedding = jnp.array([1.0, 1.0, 1.0])


def sample_jax(iterations: int):
    print(f"Running JAX with {iterations}")

    start = time.time()
    solver.integrate(
        q0=jnp.array([1.0]),
        pi0=jnp.array([1.0]),
        t0=0,
        iterations=iterations,
        additional_data=embedding
    )
    end = time.time()
    return end - start


def sample_original(iterations: int):
    start = time.time()
    dho.solve(
        embedding,
        jnp.array([1.0]), jnp.array([1.0])
    )
    end = time.time()
    return end - start


# Warm up the JIT
sample_jax(5)

# iteration_points = np.linspace(10, 1000, 100, dtype=int)
iteration_points = 10 ** np.arange(0, 7, 0.25)
times = np.array([sample_jax(int(n)) for n in iteration_points])

df = pandas.DataFrame({
    "iterations": iteration_points,
    "times": times
})

project_data_root.joinpath('figures').mkdir(
    exist_ok=True,
    parents=True
)

current_datetime = datetime.datetime.now().isoformat()

df.to_csv(project_data_root.joinpath(
    'figures',
    f"{current_datetime}.csv"
))

plt.plot(iteration_points, times)
plt.xlabel("Number of iterations")
plt.ylabel("Total integration time / s")

plt.savefig(project_data_root.joinpath(
    'figures',
    f"{current_datetime}.png"
))
plt.show()
