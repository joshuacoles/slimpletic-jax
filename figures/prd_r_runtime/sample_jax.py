import time
import datetime

import jax

from neural_networks.data import project_data_root
import numpy as np
import jax.numpy as jnp

from neural_networks.data.families import dho
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle
import figures.orbit_util as orbit

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

# Parameters
G = 39.478758435  # (in AU^3/M_sun/yr^2))
M_Sun = 1.0  # (in solar masses)
rho = 2.0  # (in g/cm^3)
d = 5.0e-3  # (in cm)
beta = 0.0576906 * (2.0 / rho) * (1.0e-3 / d)  # (dimensionless)
c = 63241.3  # (in AU/yr)
m = 1.


def lagrangian(q, v, t):
    return 0.5 * jnp.dot(v, v) + (1.0 - beta) * G * M_Sun / jnp.dot(q, q) ** 0.5


def nonconservative(qp, qm, vp, vm, t):
    a = jnp.dot(vp, qm) + jnp.dot(vp, qp) * jnp.dot(qp, qm) / jnp.dot(qp, qp)
    b = -beta * G * M_Sun / c / jnp.dot(qp, qp)
    return a * b


q0, v0 = orbit.Calc_Cartesian(1.0, 0.2, 0.0, 0.0, 0.0, 0.0, (1.0 - beta) * G * M_Sun)
pi0 = v0  # Dust taken to have unit mass


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
            lagrangian=lagrangian,
            k_potential=nonconservative,
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
