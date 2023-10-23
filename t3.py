import jax.numpy as jnp
import scipy

r = 5


def leg_poly(r: int):
    return scipy.special.legendre(r).c


poly = leg_poly(r)
differentiated_poly = jnp.polyder(poly)
real_roots_of_poly = jnp.roots(differentiated_poly)

print(real_roots_of_poly.real)
