from datetime import datetime
import math

import jax
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import jax.numpy as jnp
import numpy as np


def determine_grid_dimensions(num_plots):
    # Take the square root of the total number of plots
    root = math.sqrt(num_plots)

    # Get the lower and upper integer bounds
    lower_bound = math.floor(root)
    upper_bound = math.ceil(root)

    # If the square of the lower bound is enough to accommodate all plots, use that
    if lower_bound * lower_bound >= num_plots:
        return lower_bound, lower_bound

    # If not, check if the lower bound multiplied by the upper bound is enough
    elif lower_bound * upper_bound >= num_plots:
        return lower_bound, upper_bound

    # If all else fails, use the upper bound for both dimensions
    else:
        return upper_bound, upper_bound


def loss_plot(
        ax,
        dimension,
        loss_fn,
        embedding,
        variation_range
):
    # Extend to a 2D array with zeros in the 1st and 3rd positions
    offsets = jnp.zeros((variation_range.size, embedding.shape[0]))
    offsets = offsets.at[:, dimension].set(variation_range)

    ax.set_title(f"LF, emb[{dimension}]")
    ax.plot(
        variation_range,
        jax.vmap(loss_fn)(embedding + offsets)
    )


def plot_variation_graph(
        loss_fn,
        embedding,
        variation_range,
        fig,
        grid_spec,
        loss_variation_size
):
    for i in range(embedding.size):
        ax = fig.add_subplot(grid_spec[i // loss_variation_size[1], i % loss_variation_size[1]])
        loss_plot(
            ax,
            i,
            loss_fn,
            embedding,
            variation_range
        )


def create_plots(embedding_size, label):
    fig = plt.figure(figsize=(16, 8), layout="constrained")
    fig.suptitle(f"{label}, {datetime.now().isoformat()}")
    grid_spec = GridSpec(1, 2, figure=fig)
    loss_variation_size = determine_grid_dimensions(embedding_size)
    variation_grid_spec = GridSpecFromSubplotSpec(*loss_variation_size, grid_spec[0])
    comparison_ax = fig.add_subplot(grid_spec[0, 1])

    return fig, variation_grid_spec, comparison_ax, loss_variation_size


def plot_comparison(
        embedding,
        true_embedding,
        solve,
        comparison_ax,
        t
):
    actual_q, _actual_pi = solve(embedding)
    expected_q, _expected_pi = solve(true_embedding)
    comparison_ax.plot(t, expected_q)
    comparison_ax.plot(t, actual_q, linestyle='dashed')
