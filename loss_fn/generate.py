#!/usr/bin/env python
import os
import sys
import pathlib

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
family_key = families.power_series_with_prefactor.__name__
loss_fn_key = loss_fns.q_rms_embedding_norm_huber.__name__

system = create_system(
    family_key,
    loss_fn_key,
    system_key,
    timesteps=100
)

maxiter = 200
samples = 100

batch = datetime.datetime.now().isoformat()
data_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.joinpath('data')
root = f"{data_dir}/{loss_fn_key}/{system_key}-{family_key}/{batch}"
os.makedirs(root)

true_loss = system.loss_fn(system.true_embedding)

for i in range(samples):
    print("Running", i)
    random_initial_embedding = jnp.array(np.random.rand(system.true_embedding.size))
    gradient_descent_result = jaxopt.GradientDescent(
        system.loss_fn,
        maxiter=maxiter,
        verbose=True,
    ).run(random_initial_embedding)

    embedding = gradient_descent_result.params
    print(embedding)
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
            "family": family_key,
            "loss_fn": loss_fn_key,
        }
    }, open(f"{root}/{i}.json", "w"), indent=2)
