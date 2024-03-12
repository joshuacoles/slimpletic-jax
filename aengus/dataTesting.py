import numpy as np

DATASIZE = 20480
DATA_TIMESTEPS = 40
TRAINING_TIMESTEPS = 10

dataName = "HarmonicOscillator"
XName = "Data/" + dataName + "/xData.npy"
YName = "Data/" + dataName + "/yData.npy"


X = np.load(XName)
Y = np.load(YName)
print(X)

print(Y.shape)


