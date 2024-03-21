#!/usr/bin/env python3

import os
import sys
import pathlib

# Get the directory of the current script
script_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir.parent))

import json
import jax.numpy as jnp

from loss_fn.graph_helpers import create_plots, plot_variation_graph
from loss_fn.utils import create_system

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage {sys.argv[0]} <json file> <png output>", file=sys.stderr)
        sys.exit(1)

    json_file = sys.argv[1]
    png_out = sys.argv[2]

    data = json.load(open(json_file, 'r'))
    print(data)

    true_embedding = jnp.array(data['true_embedding'])
    embedding = jnp.array(data['found_embedding'])

    family_key = data['keys']['family']
    loss_fn_key = data['keys']['loss_fn']
    system_key = data['keys']['system']
    timesteps = data['timesteps']

    system = create_system(
        family=family_key,
        loss_fn=loss_fn_key,
        true_embedding=true_embedding,
        timesteps=timesteps
    )

    fig, variation_grid_spec, comparison_ax, loss_variation_size = create_plots(
        embedding_size=system.true_embedding.size,
        label=f"{family_key} {system_key} {loss_fn_key}"
    )

    plot_variation_graph(
        system.loss_fn,
        embedding,
        jnp.linspace(-1, 1, 100),
        fig,
        variation_grid_spec,
        loss_variation_size
    )

    target_q = system.solve(system.true_embedding)[0]
    actual_q = system.solve(embedding)[0]
    comparison_ax.plot(system.t, target_q, label="Expected")
    comparison_ax.plot(system.t, actual_q, label="Predicted", linestyle="--")
    comparison_ax.legend()

    fig.savefig(png_out)
