import jax.lax

from slimpletic import SolverScan, DiscretisedSystem, SolverBatchedScan, SolverManual, GGLBundle
import jax.numpy as jnp

power_series_order = 2


def function(q, v, _, an):
    return jax.lax.fori_loop(
        0, power_series_order,
        lambda i, acc: acc + an[2 * i] * q[0] ** (i + 1),
        0.0
    ) + jax.lax.fori_loop(
        0, power_series_order,
        lambda i, acc: acc + an[2 * i + 1] * v[0] ** (i + 1),
        0.0
    )


system = DiscretisedSystem(
    ggl_bundle=GGLBundle(r=0),
    dt=0.01,
    lagrangian=function,
    k_potential=None,
    pass_additional_data=True,
)

solver = SolverScan(system)

solver.integrate(
    q0=jnp.array([0.0]),
    pi0=jnp.array([1.0]),
    t0=0,
    iterations=1000,
    additional_data=jnp.array(an),
    result_orientation='coordinate'
)
