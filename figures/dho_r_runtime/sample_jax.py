import time
import datetime

import jax

from neural_networks.data import project_data_root
import numpy as np
import jax.numpy as jnp

from neural_networks.data.families import dho
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle

current_datetime = datetime.datetime.now().isoformat()
output_dir = project_data_root / 'figures' / 'dho_r_runtime' / current_datetime
output_dir.mkdir(exist_ok=True, parents=True)

repeat_samples = 4

fig_out = output_dir.joinpath('figure.png')
jax_csv_out = open(output_dir.joinpath('jax.csv'), "w")
jax_csv_out.write("r,computation_time,jit_time\n")

dt = 0.1
iterations = 500

# Run the system once to warm up the JIT
embedding = jnp.array([1.0, 1.0, 1.0])


def sample_jax(r: int):
    print(f"Running JAX with r={r}")

    start_setup = time.time()

    for _ in range(repeat_samples):
        jax.clear_caches()

        ggl_bundle = GGLBundle(r=r)
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

    jit_time = (time.time() - start_setup) / repeat_samples

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
    jax_csv_out.write(f"{r},{computation_time},{jit_time}\n")
    jax_csv_out.flush()
    return computation_time, jit_time


r_values = jnp.arange(10, 20, 1)
jax_times = np.array([sample_jax(int(n)) for n in r_values])
jax_csv_out.close()
