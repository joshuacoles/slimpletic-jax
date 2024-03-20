import dataclasses
import json
from typing import Callable

import numpy as np
from jax import numpy as jnp, jit

from slimpletic import GGLBundle, SolverScan, DiscretisedSystem


def make_solver(family, iterations):
    q0 = jnp.array([0.0])
    pi0 = jnp.array([1.0])
    t0 = 0
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
    timesteps: int


def create_system(family_key, loss_fn_key, system_key_or_true_embedding, timesteps):
    import families
    import loss_fns
    import pathlib

    system_manifest = json.load(open(pathlib.Path(__file__).parent.joinpath('systems.json'), "r"))

    if isinstance(system_key_or_true_embedding, str):
        true_embedding = jnp.array(system_manifest[family_key][system_key_or_true_embedding])
    else:
        true_embedding = system_key_or_true_embedding

    family = getattr(families, family_key)
    t, solve = make_solver(family, timesteps)
    loss_fn = getattr(loss_fns, loss_fn_key)(solve, true_embedding)

    return System(
        family=jit(family),
        loss_fn=jit(loss_fn),
        true_embedding=true_embedding,
        t=t,
        solve=solve,
        timesteps=timesteps,
    )
