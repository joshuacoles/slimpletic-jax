#!/usr/bin/env python
import datetime
import json
import os
import sys
import pathlib
from tqdm import tqdm

# When running it from the command line we need to add the required things to the path
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(str(script_dir.parent))

import jaxopt
from jax import numpy as jnp
import numpy as np

from ..kinds import families, loss_fns, systems
from ..utils import create_system

# import logging
# Logging, show all logs
# logging.basicConfig(level=logging.DEBUG)

args = sys.argv
if len(args) > 1:
    system_key = args[1]
    loss_fn_key = args[2]
    samples = int(args[3])
    maxiter = int(args[4])
else:
    system_key = systems.shm_prefactor
    loss_fn_key = loss_fns.q_rms_huber_embedding_norm
    samples = 5
    maxiter = 200

system = create_system(
    physical_system=system_key,
    loss_fn=loss_fn_key,
    timesteps=100
)

verbose = False

batch = datetime.datetime.now().isoformat()
data_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent.joinpath('data')
root = f"{data_dir}/{system.loss_fn_key}/{system.physical_system.key}/{batch}"
os.makedirs(root)

true_loss = system.loss_fn(system.true_embedding)
gradient_descent = jaxopt.GradientDescent(system.loss_fn, maxiter=maxiter, verbose=verbose)

for i in tqdm(range(samples)):
    random_initial_embedding = jnp.array(np.random.rand(system.true_embedding.size))
    gradient_descent_result = gradient_descent.run(random_initial_embedding)

    embedding = gradient_descent_result.params
    achieved_loss = system.loss_fn(embedding)

    tqdm.write(f"Found embedding: {embedding}")
    tqdm.write(f"Loss found: {achieved_loss} vs true loss: {true_loss}")
    tqdm.write(f"Iterations: {gradient_descent_result.state.iter_num}")
    json.dump({
        "initial_embedding": random_initial_embedding.tolist(),
        "found_embedding": embedding.tolist(),
        "loss": float(achieved_loss),
        "true_loss": float(true_loss),
        "maxiter": maxiter,
        "timesteps": system.timesteps,
        "opt_state": {
            "iter_num": gradient_descent_result.state.iter_num.tolist(),
        },
        "keys": {
            "system": system.physical_system.to_json(),
            "loss_fn": system.loss_fn_key,
        },
    }, open(f"{root}/{i}.json", "w"), indent=2)
