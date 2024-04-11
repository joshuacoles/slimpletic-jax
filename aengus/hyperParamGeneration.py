import numpy as np
import pickle

# default lists
reg_all = [1, 1, 1, 1]
reg_first = [1, 0, 0, 0]
reg_last = [0, 0, 0, 1]

units_const = [5, 5, 5, 5]
units_lin = [20, 15, 10, 5]
units_exp = [50, 25, 10, 5]

Layers = [1, 2, 3, 4]
BatchSize = [64, 128, 256]
Dropout = [0.05, 0.1, 0.2, 0.3, 0.4]
regulariser = [reg_all, reg_last, reg_first]
units = [units_const, units_lin, units_exp]



def checkList(big, small):
    for element in big:
        if np.array_equal(small, element):
            return False
    return True


permutations = []

for layer in Layers:
    for size in BatchSize:
        for prob in Dropout:
            if layer == 1:
                newlist = [layer, size, prob, regulariser[0], units[0]]
                permutations.append(newlist)
            else:
                for reglist in regulariser:
                    for unitlist in units:
                        newlist = [layer, size, prob, reglist, unitlist]
                        permutations.append(newlist)

print(permutations)
with open('HyperParams.pkl', 'wb') as f:
    pickle.dump(permutations, f)
f.close()


