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

    family_label = data['keys']['family']
    system_label = data['keys']['system']
    loss_fn_label = data['keys']['loss_fn']

    system = create_system(family_label, loss_fn_label, true_embedding)

    fig, variation_grid_spec, comparison_ax, loss_variation_size = create_plots(
        embedding_size=true_embedding.size,
        label=f"{family_label} {system_label} {loss_fn_label}"
    )

    plot_variation_graph(
        system.loss_fn,
        embedding,
        jnp.linspace(-1, 1, 100),
        fig,
        variation_grid_spec,
        loss_variation_size
    )

    target_q = system.solve(true_embedding)[0]
    actual_q = system.solve(embedding)[0]
    comparison_ax.plot(system.t, target_q, label="Expected")
    comparison_ax.plot(system.t, actual_q, label="Predicted", linestyle="--")
    comparison_ax.legend()

    fig.savefig(png_out)
