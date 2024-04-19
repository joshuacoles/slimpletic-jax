import time
import datetime

import jax

from neural_networks.data import project_data_root
import original

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from neural_networks.data.families import dho
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle

r = 2
dt = 0.1

ggl_bundle = GGLBundle(r=r)
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

repeat_samples = 4

fig_out = output_dir.joinpath('figure.png')
jax_csv_out = open(output_dir.joinpath('jax.csv'), "w")
original_csv_out = open(output_dir.joinpath('original.csv'), "w")

# Run the system once to warm up the JIT
embedding = jnp.array([1.0, 1.0, 1.0])
original = original.dho(1.0, 1.0, 1.0, r)


def sample_original(iterations: int):
    print(f"Running original with {iterations}")
    start = time.time()
    for _ in range(repeat_samples):
        original(iterations)

    time_elapsed = (time.time() - start) / repeat_samples
    original_csv_out.write(f"{iterations},{time_elapsed}\n")
    original_csv_out.flush()

    return time_elapsed


def sample_jax(iterations: int):
    print(f"Running JAX with {iterations}")

    start_jit = time.time()

    for _ in range(repeat_samples):
        solver.integrate._clear_cache()
        print("Cache going into jit", solver.integrate._cache_size())
        solver.integrate(
            q0=jnp.array([1.0]),
            pi0=jnp.array([1.0]),
            t0=0,
            iterations=iterations,
            additional_data=embedding
        )

    jit_time = (time.time() - start_jit) / repeat_samples

    print("Cache going into comp", solver.integrate._cache_size())
    start_computation = time.time()

    for _ in range(repeat_samples):
        solver.integrate(
            q0=jnp.array([1.0]),
            pi0=jnp.array([1.0]),
            t0=0,
            iterations=iterations,
            additional_data=embedding
        )

    computation_time = (time.time() - start_computation) / repeat_samples

    # Write data eagerly to save stuff for if we have to early exit
    jax_csv_out.write(f"{iterations},{computation_time},{jit_time}\n")
    jax_csv_out.flush()
    return computation_time, jit_time


iteration_points = 2 ** np.arange(1, 6 * 3, 6)
jax_times = np.array([sample_jax(int(n)) for n in iteration_points])

jax_comp_times = jax_times[:, 0]
jax_jit_times = jax_times[:, 1]

original_iteration_points = np.array([10, 100, 200, 500, 1000])
original_times = np.array([sample_original(int(n)) for n in original_iteration_points])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    iteration_points,
    jax_comp_times,
    label="Computation time",
)

ax.plot(
    iteration_points,
    jax_jit_times,
    label="JIT time",
)

ax.plot(
    iteration_points,
    jax_jit_times,
    label="JIT time",
)

ax.plot(
    original_iteration_points,
    original_times,
    label="Original",
)

ax.set_xlabel("Number of iterations")
ax.set_ylabel("Total integration time / s")

ax.set_xscale('log')
fig.legend()
fig.savefig(fig_out)

plt.show()

jax_csv_out.close()
original_csv_out.close()
