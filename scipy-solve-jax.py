import scipy
from jax import numpy as jnp, jit, grad


def function(x):
    return x ** 2 + 2 * x + 1


def implicit_solve(f, x0):
    return scipy.optimize.root(f, x0, jac=grad(f))


f = jit(function)

print(implicit_solve(f, 1.0))
print(implicit_solve(f, 1.01))
print(implicit_solve(f, 10))
print(implicit_solve(f, 10.22))
