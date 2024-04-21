from .helpers import jax_enable_x64
from .solver import DiscretisedSystem, SolverBatchedScan, SolverManual, SolverScan
from .ggl import GGLBundle

jax_enable_x64()

__all__ = [
    'GGLBundle',
    'DiscretisedSystem',
    'SolverBatchedScan',
    'SolverManual',
    'SolverScan',
    'make_solver'
]


def make_solver(r, lagrangian, k_potential, dt):
    return SolverScan(
        DiscretisedSystem(
            dt=dt,
            ggl_bundle=GGLBundle(r=r),
            lagrangian=lagrangian,
            k_potential=k_potential
        )
    )
