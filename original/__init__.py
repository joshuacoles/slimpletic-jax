from .slimplectic import GalerkinGaussLobatto

__all__ = ['GalerkinGaussLobatto', 'dho']


def dho(m, k, ll, r):
    import numpy as np

    sys = slimplectic.GalerkinGaussLobatto('t', ['q'], ['v'])
    L = 0.5 * m * np.dot(sys.v, sys.v) - 0.5 * k * np.dot(sys.q, sys.q)
    K = -ll * np.dot(sys.vp, sys.qm)

    sys.discretize(L, K, r, method='explicit', verbose=True)

    def call(iterations):
        dt = 0.1 * np.sqrt(m / k)
        tmax = iterations * np.sqrt(m / k)
        t = dt * np.arange(0, int(tmax / dt) + 1)

        # Initial data (at t=0)
        q0 = [1.]
        pi0 = [0.25 * dt * k]

        # Integrate the system
        sys.integrate(q0, pi0, t, dt)

    return call
