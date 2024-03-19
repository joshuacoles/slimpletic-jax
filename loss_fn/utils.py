import dataclasses
from typing import Callable

import numpy as np
from jax import numpy as jnp

from slimpletic import GGLBundle, SolverScan, DiscretisedSystem


def make_solver(family):
    q0 = jnp.array([0.0])
    pi0 = jnp.array([1.0])
    t0 = 0
    iterations = 100
    dt = 0.1
    t = t0 + dt * np.arange(0, iterations + 1)

    ggl_bundle = GGLBundle(r=2)

    # The system which will be used when computing the loss function.
    embedded_system_solver = SolverScan(DiscretisedSystem(
        dt=dt,
        ggl_bundle=ggl_bundle,
        lagrangian=family,
        k_potential=None,
        pass_additional_data=True
    ))

    return t, lambda embedding: embedded_system_solver.integrate(
        q0=q0,
        pi0=pi0,
        t0=t0,
        iterations=iterations,
        additional_data=embedding
    )


@dataclasses.dataclass
class System:
    family: Callable
    loss_fn: Callable
    true_embedding: jnp.ndarray
    t: jnp.ndarray
    solve: Callable


def create_system(family_key, loss_fn_key, true_embedding):
    import families
    import loss_fns

    family = getattr(families, family_key)
    t, solve = make_solver(family)
    loss_fn = getattr(loss_fns, loss_fn_key)(solve, true_embedding)

    return System(
        family=family,
        loss_fn=loss_fn,
        true_embedding=true_embedding,
        t=t,
        solve=solve
    )
