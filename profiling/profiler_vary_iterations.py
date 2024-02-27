import time

import matplotlib.pyplot as plt

from original_solver import original
from slimpletic import SolverScan, DiscretisedSystem, SolverBatchedScan, SolverManual, GGLBundle
import jax.numpy as jnp
import numpy as np


# System Setup
def lagrangian(q, v, t):
    return 0.5 * v ** 2 - 0.5 * q ** 2


q0 = jnp.array([0.0])
pi0 = jnp.array([1.0])
t0 = 0
iterations = 1000
dt = 0.1
dof = 1
t = t0 + dt * np.arange(0, iterations + 1)
ggl_bundle = GGLBundle(r=0)

r = 0
m = 1.0
k = 1.0
ll = 0.5 * np.sqrt(m * k)


def lagrangian_f(q, v, t):
    return 0.5 * m * jnp.dot(v, v) - 0.5 * k * jnp.dot(q, q)


def k_potential_f(qp, qm, vp, vm, t):
    return -ll * jnp.dot(vp, qm)


system = DiscretisedSystem(
    ggl_bundle=GGLBundle(r=r),
    dt=dt,
    lagrangian=lagrangian_f,
    k_potential=k_potential_f,
)

solver_batched_scan = SolverBatchedScan(system, batch_size=100)
solver_manual = SolverManual(system)
solver_scan = SolverScan(system)


def perform_sampling(samples, work):
    sample_durations = []
    for i in range(samples):
        start_time = time.time_ns()
        work()
        duration = time.time_ns() - start_time
        sample_durations.append(duration)
    return sample_durations


samples = 200

solver_scan_by_iterations = []
original_by_iterations = []
solver_batched_by_iterations = []

iterations_list = range(100, 1000, 100)

for iterations in iterations_list:
    sovler_scan_samples = perform_sampling(
        samples,
        lambda: solver_scan.integrate(
            q0=jnp.array(np.random.normal(size=dof) * 10),
            pi0=jnp.array(np.random.normal(size=dof) * 10),
            t0=np.random.normal() * 100,
            iterations=iterations
        )
    )

    solver_batched_scan_samples = perform_sampling(
        samples,
        lambda: solver_batched_scan.integrate(
            q0=jnp.array(np.random.normal(size=dof) * 10),
            pi0=jnp.array(np.random.normal(size=dof) * 10),
            t0=np.random.normal() * 100,
            iterations=iterations
        )
    )

    original_samples = perform_sampling(
        samples,
        lambda: original.integrate(
            q0=np.array(q0),
            pi0=np.array(pi0),
            t=t0 + dt * np.arange(0, iterations + 1),
            dt=dt,
        )
    )

    solver_scan_by_iterations.append(np.mean(sovler_scan_samples[5:]))
    original_by_iterations.append(np.mean(original_samples[5:]))
    solver_batched_by_iterations.append(np.mean(solver_batched_scan_samples[5:]))

plt.plot(iterations_list, np.log(np.array(solver_scan_by_iterations)), label="scan")
plt.plot(iterations_list, np.log(np.array(original_by_iterations)), label="original")
plt.plot(iterations_list, np.log(np.array(solver_batched_by_iterations)), label=f"batched batch={solver_batched_scan.batch_size}")
plt.xlabel("Iterations")
plt.ylabel("log duration (ns)")
plt.legend()
plt.show()