import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures import figures_root
from neural_networks.data import project_data_root
from scipy.optimize import curve_fit

iter_t_jax = pd.read_csv(project_data_root / "dho_n_t/jax.csv")
iter_t_original = pd.read_csv(project_data_root / "dho_n_t/original.csv")


def linear(x, a, b):
    return a * x + b

log(iter_t_jax.computation_time) = a * log(iter_t_jax.iterations) + b

mask = iter_t_jax.iterations > 10 ** 5
p, pcov = curve_fit(linear, np.log(iter_t_jax.iterations[mask]), np.log(iter_t_jax.computation_time[mask]))
print("comp v", p)
print("comp std", np.sqrt(np.diag(pcov)))
