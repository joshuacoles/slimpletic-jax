import matplotlib.pyplot as plt
from matplotlib import gridspec

import jax
import jax.numpy as jnp
import neural_networks.data.families as families
from neural_networks.data.generate_data_impl import setup_solver

# region System

iterations = 100
q0 = jnp.array([0.0])
pi0 = jnp.array([1.0])
t0 = 0
dt = 0.1
t = t0 + dt * jnp.arange(0, iterations + 1)

family = families.dho
solver = setup_solver(
    family=family,
    iterations=iterations,
)

true_embedding = jnp.array([2.0, 3.0, 1.0])
true_q, true_pi = solver(
    true_embedding,
    q0,
    pi0
)


# %%

def loss_fn(embedding):
    q_weight = 0.5
    pi_weight = 0.5

    a, b = solver(
        embedding,
        q0,
        pi0
    )

    rms = jnp.sqrt(jnp.mean(q_weight * (a - true_q) ** 2 + pi_weight * (b - true_pi) ** 2))
    non_negatives = jnp.mean(jax.lax.select(embedding < 0, jnp.exp(-10 * embedding), jnp.zeros_like(embedding)))

    non_neg = False
    if non_neg:
        return rms + non_negatives
    else:
        return rms


vmap = jax.vmap(loss_fn)

# endregion

# %%

# Create a figure and axes
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(2, 6)


# axes = [fig.add_subplot(g) for g in gs]


def plot_dho3_close():
    close_variation_range = jnp.linspace(-1, 1, 200)

    labels = [
        "$\\Delta m$",
        "$\\Delta k$",
        "$\\Delta \\lambda$"
    ]

    positions = [
        (0, 0),
        (0, 1),
        (1, 0)
    ]

    for i in range(0, 3):
        ax = fig.add_subplot(gs[positions[i]])
        var_vec = jnp.zeros((close_variation_range.size, 3)).at[:, i].set(close_variation_range)

        ax.set_title(labels[i])
        ax.plot(
            close_variation_range,
            vmap(true_embedding + var_vec),
        )


def plot_dho3_far():
    far_variation_range = jnp.linspace(-4, 4, 200)
    mass_variation_range = jnp.linspace(-4, 4, 200)
    m_var = jnp.zeros((mass_variation_range.size, 3)).at[:, 0].set(mass_variation_range)
    k_var = jnp.zeros((far_variation_range.size, 3)).at[:, 1].set(far_variation_range)
    l_var = jnp.zeros((far_variation_range.size, 3)).at[:, 2].set(far_variation_range)

    m_plot = fig.add_subplot(gs[(0, 2)])
    m_plot.set_title("$\\Delta m$")
    m_plot.plot(mass_variation_range, vmap(true_embedding + m_var))
    m_plot.set_xticks([-4, -2, 0, 2, 4])

    k_plot = fig.add_subplot(gs[(0, 3)])
    k_plot.set_title("$\\Delta k$")
    k_plot.plot(far_variation_range, vmap(true_embedding + k_var))
    m_plot.set_xticks([-4, -2, 0, 2, 4])

    l_plot = fig.add_subplot(gs[(1, 2)])
    l_plot.set_title("$\\Delta \\lambda$")
    l_plot.plot(far_variation_range, vmap(true_embedding + l_var))
    m_plot.set_xticks([-4, -2, 0, 2, 4])


def plot_dho4():
    family = families.dho_prefactor
    solver = setup_solver(
        family=family,
        iterations=iterations,
    )

    true_embedding = jnp.array([1.0, 3.0, 1.0, 1.0])
    true_q, true_pi = solver(
        true_embedding,
        q0,
        pi0
    )

    def loss_fn(embedding):
        q_weight = 0.5
        pi_weight = 0.5

        a, b = solver(
            embedding,
            q0,
            pi0
        )

        rms = jnp.sqrt(jnp.mean(q_weight * (a - true_q) ** 2 + pi_weight * (b - true_pi) ** 2))
        non_negatives = jnp.mean(jax.lax.select(embedding < 0, jnp.exp(-10 * embedding), jnp.zeros_like(embedding)))

        non_neg = False
        if non_neg:
            return rms + non_negatives
        else:
            return rms

    vmap = jax.vmap(loss_fn)

    close_variation_range = jnp.linspace(-1, 1, 200)

    labels = [
        "$\\Delta m$",
        "$\\Delta k$",
        "$\\Delta \\lambda$",
        "$\\Delta \\alpha$"
    ]

    positions = [
        (0, 4),
        (0, 5),
        (1, 4),
        (1, 5),
    ]

    for i in range(0, 4):
        ax = fig.add_subplot(gs[positions[i]])
        var_vec = jnp.zeros((close_variation_range.size, 4)).at[:, i].set(close_variation_range)

        ax.set_title(labels[i])
        ax.plot(
            close_variation_range,
            vmap(true_embedding + var_vec),
        )


plot_dho3_close()
plot_dho3_far()
plot_dho4()

# Add titles for each 2x2 section
fig.text(1. / 6, 0.95, 'DHO 3, close to minima', ha='center', fontsize=12)
fig.text(3. / 6, 0.95, 'DHO 3, far to minima', ha='center', fontsize=12)
fig.text(5. / 6, 0.95, 'DHO 4, close to minima', ha='center', fontsize=12)

# Adjust the spacing between subplots
plt.tight_layout(pad=0.5)

# Add additional padding at the top
plt.subplots_adjust(top=0.85)

# Display the plot
plt.savefig('loss-function-behaviour.pdf')
plt.show()
