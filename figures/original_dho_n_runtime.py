import time
import datetime

from neural_networks.data import project_data_root

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from neural_networks.data.families import dho
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

current_datetime = datetime.datetime.now().isoformat()
output_dir = project_data_root.joinpath('figures', 'dho_n_runtime', current_datetime)
output_dir.mkdir(exist_ok=True, parents=True)

fig_out = output_dir.joinpath('figure.png')
csv_out = open(output_dir.joinpath('data.csv'), "w")

# Run the system once to warm up the JIT
embedding = jnp.array([1.0, 1.0, 1.0])


def sample(iterations: int):
    print(f"Running JAX with {iterations}")

    start = time.time()
    repeat_samples = 4

    for _ in range(repeat_samples):
        solver.integrate(
            q0=jnp.array([1.0]),
            pi0=jnp.array([1.0]),
            t0=0,
            iterations=iterations,
            additional_data=embedding
        )

    end = time.time()
    end_start = (end - start) / repeat_samples

    # Write data eagerly to save stuff for if we have to early exit
    csv_out.write(f"{iterations},{end_start}\n")
    csv_out.flush()
    return end_start


# Warm up the JIT
sample(5)

# iteration_points = np.linspace(10, 1000, 100, dtype=int)
iteration_points = 10 ** np.arange(0, 7, 0.1)
times = np.array([sample(int(n)) for n in iteration_points])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = ax.plot(
    iteration_points,
    times,
)

ax.set_xlabel("Number of iterations")
ax.set_ylabel("Total integration time / s")

ax.set_xscale('log')
fig.savefig(fig_out)

plt.show()
