import dataclasses
import json
from typing import Callable, Union

import numpy as np
from jax import numpy as jnp, jit

from loss_fn.families import Family
from slimpletic import GGLBundle, SolverScan, DiscretisedSystem


def make_solver(family: Family, iterations: int):
    q0 = jnp.array([0.0])
    pi0 = jnp.array([1.0])
    t0 = 0
    dt = 0.1
    t = t0 + dt * np.arange(0, iterations + 1)

    ggl_bundle = GGLBundle(r=2)

    if family.k_potential is None:
        k_potential = None
    else:
        k_potential = jit(family.k_potential)

    # The system which will be used when computing the loss function.
    embedded_system_solver = SolverScan(DiscretisedSystem(
        dt=dt,
        ggl_bundle=ggl_bundle,
        lagrangian=jit(family.lagrangian),
        k_potential=k_potential,
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
    family: Family
    loss_fn: Callable
    loss_fn_key: str
    true_embedding: jnp.ndarray
    t: jnp.ndarray
    solve: Callable
    timesteps: int


def create_system(family: Union[Family, str], loss_fn: Union[Callable, str],
                  system_key_or_true_embedding: Union[str, jnp.ndarray], timesteps: int):
    import families
    import loss_fns
    import pathlib

    system_manifest = json.load(open(pathlib.Path(__file__).parent.joinpath('systems.json'), "r"))

    if isinstance(family, str):
        family = getattr(families, family)

    if isinstance(loss_fn, str):
        loss_fn = getattr(loss_fns, loss_fn)

    if isinstance(system_key_or_true_embedding, str):
        true_embedding = jnp.array(system_manifest[family.key][system_key_or_true_embedding])
    else:
        true_embedding = system_key_or_true_embedding

    t, solve = make_solver(family, timesteps)
    loss_fn = loss_fn(solve, true_embedding)

    return System(
        family=family,
        loss_fn=jit(loss_fn),
        loss_fn_key=loss_fn.__name__,
        true_embedding=true_embedding,
        t=t,
        solve=solve,
        timesteps=timesteps,
    )


def create_system_from_fns(family, loss_fn, true_embedding, timesteps):
    t, solve = make_solver(family, timesteps)
    loss_fn = loss_fn(solve, true_embedding)

    return System(
        family=family,
        loss_fn=jit(loss_fn),
        true_embedding=true_embedding,
        t=t,
        solve=solve,
        timesteps=timesteps
    )
