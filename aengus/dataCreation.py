import numpy as np
import os
from .MVP_Data_Creation_Josh import slimplecticSoln

DATASIZE = 20480
TIMESTEPS = 40

suffix = "Data/" + "HarmonicOscillator"


def genData():
    q_data, pi_data, L_data = [], [], []

    for _ in range(DATASIZE):
        if _ % 100 == 0:
            print(_)
        if _ <= DATASIZE / 2:
            q, p, l = slimplecticSoln(TIMESTEPS, None)
        else:
            q, p, l = slimplecticSoln(TIMESTEPS, None)
        q_data.append(q[0])
        pi_data.append(p[0])
        L_data.append(l)

    X = np.array([q_data, pi_data]).reshape((DATASIZE, TIMESTEPS + 1, 2))
    Y = np.array(L_data)
    return X, Y


X, Y = genData()

if not os.path.exists(suffix):
    os.makedirs(suffix)

xString = suffix + "/xData"
yString = suffix + "/yData"

np.save(xString, X)
np.save(yString, Y)
