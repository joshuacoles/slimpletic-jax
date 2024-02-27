import numpy as np
from MVP_Data_Creation_Josh import slimplecticSoln
import matplotlib.pyplot as plt

DATASIZE = 20480
TIMESTEPS = 40


def genData():
    q_data, pi_data, L_data = [], [], []

    for _ in range(DATASIZE):
        if _%100 == 0:
            print(_)
        q, p, l = slimplecticSoln(TIMESTEPS)
        q_data.append(q[0])
        pi_data.append(p[0])
        L_data.append(l)

    X = np.array([q_data, pi_data]).reshape((DATASIZE,TIMESTEPS+1,2))
    Y = np.array(L_data)
    return X, Y

X,Y = genData()

np.save("xData_lowNoise",X)
np.save("yData_lowNoise",Y)