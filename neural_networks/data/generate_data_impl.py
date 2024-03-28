import os
from pathlib import Path
from typing import Callable

import numpy as np
import jax.numpy as jnp

from neural_networks.data.families import Family
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle

# The root directory where the data will be stored, $PROJECT_ROOT/data/nn_data
data_root = Path(__file__).parent.parent.parent.joinpath('data/nn_data')


def setup_solver(family: Family, iterations: int) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
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

    def solve(embedding, q0, pi0):
        print("PANDAS Recompiled!!")
        return embedded_system_solver.integrate(
            q0=q0,
            pi0=pi0,
            t0=t0,
            iterations=iterations,
            additional_data=embedding
        )

    return solve