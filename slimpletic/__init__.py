from .helpers import jax_enable_x64
from .solver_class import Solver

jax_enable_x64()

__all__ = [
    'Solver'
]
