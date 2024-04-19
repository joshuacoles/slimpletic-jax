import pandas as pd
import matplotlib.pyplot as plt

jax = pd.read_csv("../data/dho_n_t/jax.csv")
original = pd.read_csv("../data/dho_n_t/original.csv")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    jax.iterations,
    jax.computation_time,
    label="Computation time",
)

ax.plot(
    jax.iterations,
    jax.jit_time,
    label="JIT time",
)

ax.plot(
    original.iterations,
    original.time,
    label="Original",
)

ax.set_xlabel("Number of iterations")
ax.set_ylabel("Total integration time / s")

ax.set_xscale('log')
ax.set_yscale('log')
fig.legend()

plt.show()
