import time
import datetime

from neural_networks.data import project_data_root
import original

import numpy as np
import jax.numpy as jnp

r = 2
dt = 0.1

current_datetime = datetime.datetime.now().isoformat()
output_dir = project_data_root.joinpath('figures', 'dho_n_runtime', current_datetime)
output_dir.mkdir(exist_ok=True, parents=True)

repeat_samples = 4

fig_out = output_dir.joinpath('figure.png')
jax_csv_out = open(output_dir.joinpath('jax.csv'), "w")
original_csv_out = open(output_dir.joinpath('original.csv'), "w")

jax_csv_out.write("iterations,computation_time,jit_time\n")
original_csv_out.write("iterations,time\n")

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


original_iteration_points = 2 ** np.arange(15, 20, 1)
original_times = np.array([sample_original(int(n)) for n in original_iteration_points])
original_csv_out.close()
