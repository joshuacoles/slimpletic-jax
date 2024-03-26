import dataclasses
from typing import Callable, Union

import numpy as np
from jax import numpy as jnp, jit

from slimpletic import GGLBundle, SolverScan, DiscretisedSystem

from ..kinds.families import Family
from ..kinds.systems import PhysicalSystem, lookup_system
from ..kinds.loss_fns import lookup_loss_fn


def make_solver(system: PhysicalSystem, iterations: int):
    family = system.family

    q0 = system.q0
    pi0 = system.pi0
    t0 = 0
    dt = 0.1
    t = t0 + dt * np.arange(0, iterations + 1)

    ggl_bundle = GGLBundle(r=2)

    if family.k_potential is None:
        k_potential = None
    else:
        k_potential = family.k_potential

    # The system which will be used when computing the loss function.
    embedded_system_solver = SolverScan(DiscretisedSystem(
        dt=dt,
        ggl_bundle=ggl_bundle,
        lagrangian=family.lagrangian,
        k_potential=k_potential,
        pass_additional_data=True
    ))

    def solve(embedding):
        print("PANDAS Recompiled!!")
        return embedded_system_solver.integrate(
            q0=q0,
            pi0=pi0,
            t0=t0,
            iterations=iterations,
            additional_data=embedding
        )

    return t, solve


@dataclasses.dataclass
class GradientDescentBundle:
    physical_system: PhysicalSystem
    family: Family
    loss_fn: Callable
    loss_fn_key: str
    true_embedding: jnp.ndarray
    t: jnp.ndarray
    solve: Callable
    timesteps: int


def create_system(
        physical_system: Union[PhysicalSystem, dict, str],
        loss_fn: Union[Callable, dict, str],
        timesteps: int
):
    if isinstance(physical_system, str):
        physical_system = lookup_system(physical_system)
    elif isinstance(physical_system, dict):
        physical_system = PhysicalSystem.from_json(physical_system)
    elif not isinstance(physical_system, PhysicalSystem):
        raise

    t, solve = make_solver(physical_system, timesteps)

    if isinstance(loss_fn, str):
        make_loss_fn = lookup_loss_fn(loss_fn)
        loss_fn = make_loss_fn(solve, physical_system.true_embedding)
    elif isinstance(loss_fn, dict):
        make_loss_fn = lookup_loss_fn(loss_fn['key'])
        loss_fn = make_loss_fn(solve, physical_system.true_embedding, loss_fn['config'])
    elif isinstance(loss_fn, Callable):
        loss_fn = loss_fn(solve, physical_system.true_embedding)

    return GradientDescentBundle(
        physical_system=physical_system,
        family=physical_system.family,
        loss_fn=jit(loss_fn),
        loss_fn_key=loss_fn.__name__,
        true_embedding=physical_system.true_embedding,
        t=t,
        solve=solve,
        timesteps=timesteps,
    )
