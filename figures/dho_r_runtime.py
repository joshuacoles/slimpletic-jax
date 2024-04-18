import time
import datetime

from neural_networks.data import project_data_root

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import pandas

from neural_networks.data.families import Family, dho
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle

dt = 1e-15


# Run the system once to warm up the JIT
embedding = jnp.array([1.0, 1.0, 1.0])


def sample_jax(iterations: int, rvalue: int):
    print(f"Running JAX with r={rvalue}")


    ggl_bundle = GGLBundle(r=rvalue)

    start = time.time()
    # The system which will be used when computing the loss function.
    solver = SolverScan(DiscretisedSystem(
        dt=dt,
        ggl_bundle=ggl_bundle,
        lagrangian=dho.lagrangian,
        k_potential=dho.k_potential,
        pass_additional_data=True
    ))

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
sample_jax(5,0)

# iteration_points = np.linspace(10, 1000, 100, dtype=int)
r_values = [r for r in range(0,10)]
times = np.array([sample_jax(100, r) for r in r_values])

df = pandas.DataFrame({
    "rvalues": r_values,
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




fig = plt.figure()
plt.xlabel("rvalue")
plt.ylabel("Total integration time / s")
plt.plot(df.rvalues, df.times)
fig.savefig(project_data_root.joinpath(
    'figures',
    f"{current_datetime}.png"
))
plt.show()
