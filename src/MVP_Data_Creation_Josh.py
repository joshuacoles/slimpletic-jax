import random
import numpy as np
from slimpletic import SolverScan, DiscretisedSystem, GGLBundle, SolverManual
import jax
import jax.numpy as jnp

PSeriesOrder = 2


def random_coeffs(zero):
    an = []
    for i in range(0, 2 * PSeriesOrder):
        if zero:
            if (random.randint(1,5) == 1):
                an.append(0)
            else:
                an.append((random.random() - 0.5) * 20)
        else:
            an.append((random.random() - 0.5) * 20)
    return an

def HarmonicOscillatorCoeffs():
    an = []
    for i in range(0,2 * PSeriesOrder):
        if i < 2:
            an.append(0)
        else:
            an.append((random.random() - 0.5) * 20)
    return an

def function(q, v, _, an):
    return jax.lax.fori_loop(0, PSeriesOrder,
                             lambda i, acc: acc + (an[2 * i] * q[0] ** (i + 1)) + (an[2 * i + 1] * v[0] ** (i + 1)),
                             0.0)


system = DiscretisedSystem(
    ggl_bundle=GGLBundle(r=0),
    dt=0.1,
    lagrangian=function,
    k_potential=None,
    pass_additional_data=True,
)

solver = SolverManual(system)



def slimplecticSoln(timesteps,zero):
    coeffs = jnp.array(random_coeffs(zero))

    q_slim, pi_slim = solver.integrate(
        q0=jnp.array([1.0]),
        pi0=jnp.array([1.0]),
        t0=0,
        iterations=timesteps,
        additional_data=coeffs,
        result_orientation='coordinate'
    )

    # adding noise:
    q_slim += np.random.normal(0, abs(np.mean(q_slim.flatten()) / 500), np.shape(q_slim))
    pi_slim += np.random.normal(0, abs(np.mean(pi_slim.flatten()) / 500), np.shape(pi_slim))
    return q_slim, pi_slim, coeffs
