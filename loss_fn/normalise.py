import json
import os
import sys
import pathlib

import jax.numpy as jnp

# Get the directory of the current script
script_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir.parent))


def normlise_prefactor_embedding(embedding: jnp.ndarray) -> jnp.ndarray:
    first, rest = embedding[0], embedding[1:]
    return rest / first


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage {sys.argv[0]} <json file>", file=sys.stderr)
        sys.exit(1)

    json_file = sys.argv[1]
    data = json.load(open(json_file, 'r'))

    if data['keys']['family'] != 'power_series_with_prefactor':
        print("Not a power_series_with_prefactor family", file=sys.stderr)
        sys.exit(0)

    found_embedding = normlise_prefactor_embedding(jnp.array(data['found_embedding']))
    true_embedding = normlise_prefactor_embedding(jnp.array(data['true_embedding']))

    print(f"True embedding: {true_embedding}")
    print(f"Found embedding: {found_embedding}")
    print(f"Norm difference: {jnp.linalg.norm(found_embedding - true_embedding)}")
