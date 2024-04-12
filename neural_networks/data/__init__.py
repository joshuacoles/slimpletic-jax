import sys
from pathlib import Path
from typing import Union
from jax import numpy as jnp

from neural_networks.data.families import Family, lookup_family

# The root directory where all data will be stored, $PROJECT_ROOT/data
project_data_root = Path(__file__).parent.parent.parent.joinpath('data')
nn_data_root = project_data_root.joinpath("nn_data")


def nn_data_path(family_key: str, population_name: str):
    return nn_data_root.joinpath(f"{family_key}/{population_name}")


def save_nn_data(
        family: Union[Family, str],
        population_name: str,
        trajectories,
        lagrangian_embeddings
):
    family_key = family.key if isinstance(family, Family) else family

    data_dir = nn_data_path(family_key, population_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    verify_data_integrity(family, trajectories, lagrangian_embeddings)
    jnp.save(data_dir.joinpath("x"), trajectories)
    jnp.save(data_dir.joinpath("y"), lagrangian_embeddings)


def load_data(
        family: Union[Family, str],
        population_name: str
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load the data for a given family and population name.
    """
    family = family if isinstance(family, Family) else lookup_family(family)
    data_dir = nn_data_path(family.key, population_name)

    if not data_dir.exists():
        if data_dir.joinpath(family.key).exists():
            print(f"No such population {population_name} for {family.key}", file=sys.stderr)
            print(f"Existing populations: ", file=sys.stderr)
            for population_dir in data_dir.iterdir():
                print(f"\t '{population_dir.name}'", file=sys.stderr)
            raise FileNotFoundError(f"No such population {population_name} for {family.key}")
        else:
            print(f"No such family {family.key}", file=sys.stderr)
            print(f"Existing families: ", file=sys.stderr)
            for family_dir in nn_data_root.iterdir():
                print(f"\t '{family_dir.name}'", file=sys.stderr)
            raise FileNotFoundError(f"No such family {family.key}")

    x_data = jnp.load(data_dir.joinpath("x.npy"))
    y_data = jnp.load(data_dir.joinpath("y.npy"))

    verify_data_integrity(family, x_data, y_data)
    return x_data, y_data


def verify_data_integrity(family: Family, trajectories: jnp.ndarray, embeddings: jnp.ndarray):
    assert len(trajectories.shape) == 3, "Trajectories must have shape (samples, timesteps, 2 * dof)"
    assert len(embeddings.shape) == 2, "Embeddings must have shape (samples, embedding_size)"

    assert trajectories.shape[0] == embeddings.shape[0], \
        "Trajectories and embeddings must have the same number of samples"

    assert embeddings.shape[1] == family.embedding_shape[0], \
        "Embeddings must have the correct shape for the family"
