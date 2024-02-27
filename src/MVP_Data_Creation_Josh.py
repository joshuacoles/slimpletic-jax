import random
import numpy as np
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle, SolverManual
import jax
import jax.numpy as jnp

PSeriesOrder = 2

def random_coeffs():
    an = []
    for i in range(0, 2 * PSeriesOrder + 1):
        an.append((random.random() - 0.5) * 20)
    return an

def function(q, v, _, an):
    jax.debug.print("q {}", q)
    jax.debug.print("v {}", v)
    jax.debug.print("an {}", an)

    # return -jnp.dot(q, q) + jnp.dot(v, v)
    L = jax.lax.fori_loop(0, 2,
                            lambda i, acc: acc + (an[2 * i] * q[0] ** (i + 1)) + (an[2 * i + 1] * v[0] ** (i + 1)),
                            0.0) + an[-1]

    jax.debug.print("L {}", L)
    return L


def slimplecticSoln(timesteps):
    system = DiscretisedSystem(
        ggl_bundle=GGLBundle(r=0),
        dt=0.1,
        lagrangian=function,
        k_potential=None,
        pass_additional_data=True,
    )

    coeffs = jnp.array(random_coeffs())
    solver = SolverManual(system)

    q_slim, pi_slim = solver.integrate(
        q0=jnp.array([1.0]),
        pi0=jnp.array([1.0]),
        t0=0,
        iterations=timesteps,
        additional_data=coeffs,
        # result_orientation='coordinate'
    )

    # adding noise:
    q_noise = np.random.normal(0, abs(np.mean(q_slim.flatten()) / 100), np.shape(q_slim))
    pi_noise = np.random.normal(0, abs(np.mean(pi_slim.flatten()) / 100), np.shape(pi_slim))
    return q_slim + q_noise, pi_slim + pi_noise, coeffs
