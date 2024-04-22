from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from neural_networks.data import project_data_root


jax = pd.concat([
    pd.read_csv(project_data_root / "figures/dho_r_runtime/2024-04-20T15:21:31.310718/jax.csv"),
    pd.read_csv(project_data_root / "figures/dho_r_runtime/2024-04-20T16:11:07.435472/jax.csv"),
])

original = pd.concat([
    pd.read_csv(project_data_root / "figures/dho_r_runtime/2024-04-20T15:21:31.464425/original.csv"),
    pd.read_csv(project_data_root / "figures/dho_r_runtime/2024-04-20T15:58:25.658282/original.csv"),
])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    jax.r,
    jax.computation_time,
    label="JAX: computation",
)

ax.plot(
    jax.r,
    jax.jit_time,
    label="JAX: setup time",
)

ax.plot(
    original.r,
    original.time,
    label="Original: computation",
)

ax.plot(
    original.r,
    original.setup_time,
    label="Original: setup",
)

ax.set_xlabel("Order of GGL")
ax.set_ylabel("Total integration time / s")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.legend()

report_figures = Path('/Users/joshuacoles/Developer/checkouts/fyp/report/figures')
plt.savefig(report_figures / "dho_r_runtime_linear.pdf")
ax.set_yscale('log')
plt.savefig(report_figures / "dho_r_runtime.pdf")
plt.show()
