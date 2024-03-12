import random
import numpy as np

from slimplectic import GalerkinGaussLobatto

DataSetSize = 10
PSeriesOrder = 2

def random_coeffs():
    an = []
    for i in range(0, 2*PSeriesOrder + 1):
        an.append((random.random() - 0.5) * 20)
    return an


def slimplecticSoln(timesteps):
    # create object to get data from
    object = GalerkinGaussLobatto('t', ['q'], ['v'])

    # define Lagrangian
    an = random_coeffs()
    coeffs = an
    L = 0
    K = 0
    for i in range(0, PSeriesOrder):
        for qq in object.q:
            L += (an[2*i] * (qq ** (i+1)))
            K += 0 * qq
        for vv in object.v:
            L += (an[2*i + 1] * (vv ** (i+1)))
    L += an[-1]

    # Discretize the total Lagrangian using a 2nd order (r=0) explicit scheme.
    object.discretize(L, K, 0, method='explicit', verbose=True)

    # Set time samples and IC's
    dt = 0.01
    tmax = dt * timesteps
    t = dt * np.arange(0, int(tmax / dt) + 1)
    q0 = [0]
    pi0 = [random.random()]

    q_slim, pi_slim = object.integrate(q0, pi0, t)
    #adding noise:
    q_noise = np.random.normal(0, abs(np.mean(q_slim.flatten())/100), np.shape(q_slim))
    pi_noise = np.random.normal(0, abs(np.mean(pi_slim.flatten())/100), np.shape(pi_slim))
    return q_slim+q_noise, pi_slim+pi_noise, coeffs



