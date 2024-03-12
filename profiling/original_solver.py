import numpy as np

from original import GalerkinGaussLobatto

t0 = 0
iterations = 100
dt = 0.1
dof = 1

r=0
m = 1.0
k = 1.0
ll = 0.5 * np.sqrt(m * k)

original = GalerkinGaussLobatto('t', ['q'], ['v'])
L = 0.5 * m * np.dot(original.v, original.v) - 0.5 * k * np.dot(original.q, original.q)
K = -ll * np.dot(original.vp, original.qm)
original.discretize(L, K, r)
