#!/usr/bin/env python
import os
import sys
import pathlib

# Get the directory of the current script
script_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir.parent))

import datetime
import json
import os

import jaxopt
from jax import numpy as jnp, jit
import numpy as np
from loss_fn.utils import create_system

true_embedding = jnp.array([-0.5, 0.5, 0, 1.0])
family_key = "power_series_with_prefactor"
loss_fn_key = "q_only_with_embedding_norm_and_reverse_linear_weights"

system = create_system(
    family_key,
    loss_fn_key,
    true_embedding,
)

samples = 100
maxiter = 100

batch = datetime.datetime.now().isoformat()

root = f"figures/q_only_with_embedding_norm_and_reverse_linear_weights/SHM emb4/{batch}"
os.makedirs(root)

true_loss = system.loss_fn(true_embedding)

for i in range(100):
    print("Running", i)
    random_initial_embedding = jnp.array(np.random.rand(true_embedding.size))
    gradient_descent_result = jaxopt.GradientDescent(
        system.loss_fn,
        maxiter=maxiter,
        verbose=True,
    ).run(
        random_initial_embedding,
    )

    embedding = gradient_descent_result.params
    print(embedding)
    json.dump({
        "initial_embedding": random_initial_embedding.tolist(),
        "found_embedding": embedding.tolist(),
        "true_embedding": true_embedding.tolist(),
        "loss": float(system.loss_fn(embedding)),
        "true_loss": float(true_loss),
        "maxiter": maxiter,
        "opt_state": {
            "iter_num": gradient_descent_result.state.iter_num.tolist(),
        },
        "keys": {
            "system": "SHM",
            "family": family_key,
            "loss_fn": loss_fn_key,
        }
    }, open(f"{root}/{i}.json", "w"), indent=2)
