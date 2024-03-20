#!/usr/bin/env python
import os
import sys
import pathlib
from tqdm import tqdm

# When running it from the command line we need to add the required things to the path
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(str(script_dir.parent))

from loss_fn import loss_fns, families

import datetime
import json
import os

import jaxopt
from jax import numpy as jnp, jit
import numpy as np
from loss_fn.utils import create_system

system_key = "shm"

system = create_system(
    family=families.power_series_with_prefactor,
    loss_fn=loss_fns.q_rms_embedding_norm_huber,
    system_key_or_true_embedding=system_key,
    timesteps=100
)

maxiter = 200
samples = 200
verbose=False

batch = datetime.datetime.now().isoformat()
data_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.joinpath('data')
root = f"{data_dir}/{system.loss_fn_key}/{system_key}-{system.family.key}/{batch}"
os.makedirs(root)

true_loss = system.loss_fn(system.true_embedding)

for i in tqdm(range(samples)):
    random_initial_embedding = jnp.array(np.random.rand(system.true_embedding.size))
    gradient_descent_result = jaxopt.GradientDescent(
        system.loss_fn,
        maxiter=maxiter,
        verbose=verbose,
    ).run(random_initial_embedding)

    embedding = gradient_descent_result.params
    tqdm.write(f"Found embedding: {embedding}")
    json.dump({
        "initial_embedding": random_initial_embedding.tolist(),
        "found_embedding": embedding.tolist(),
        "true_embedding": system.true_embedding.tolist(),
        "loss": float(system.loss_fn(embedding)),
        "true_loss": float(true_loss),
        "maxiter": maxiter,
        "timesteps": system.timesteps,
        "opt_state": {
            "iter_num": gradient_descent_result.state.iter_num.tolist(),
        },
        "keys": {
            "system": system_key,
            "family": system.family.key,
            "loss_fn": system.loss_fn_key,
        }
    }, open(f"{root}/{i}.json", "w"), indent=2)
