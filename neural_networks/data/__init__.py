import sys
from pathlib import Path
from typing import Union

import tensorflow as tf
from jax import numpy as jnp

from neural_networks.data.families import Family, lookup_family
from neural_networks.our_code_here import family, TRAINING_TIMESTEPS, SHUFFLE_SEED

# The root directory where all data will be stored, $PROJECT_ROOT/data
project_data_root = Path(__file__).parent.parent.parent.joinpath('data')
nn_data_root = project_data_root.joinpath("nn_data")


def nn_data_path(family_key: str, population_name: str):
    return nn_data_root.joinpath(f"{family_key}/{population_name}")


def save_nn_data(
        family: Union[Family, str],
        population_name: str,
        trajectories,
        lagrangian_embeddings,
        filter_bad_data: bool = True
):
    family_key = family.key if isinstance(family, Family) else family

    data_dir = nn_data_path(family_key, population_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    if filter_bad_data:
        trajectories, lagrangian_embeddings = filter_bad_trajectories(trajectories, lagrangian_embeddings)

    verify_data_integrity(family, trajectories, lagrangian_embeddings)
    jnp.save(data_dir.joinpath("x"), trajectories)
    jnp.save(data_dir.joinpath("y"), lagrangian_embeddings)


def filter_bad_trajectories(x, y, maximum_value: int | None = None):
    """
    Filter out rows with infinite or NaN values from the data.
    """
    # Find the row indices of rows containing infinite values
    row_indices = jnp.where(jnp.any(jnp.isinf(x) | jnp.isnan(x) | x > maximum_value, axis=1))[0]

    # Create a boolean mask for rows to keep (rows without infinite values)
    x_mask = jnp.ones(x.shape[0], dtype=bool).at[row_indices].set(False)
    y_mask = jnp.ones(y.shape[0], dtype=bool).at[row_indices].set(False)

    return x[x_mask], y[y_mask]


def load_nn_data(
        family: Union[Family, str],
        population_name: str,
        filter_bad_data: bool = True,
        maximum_value: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load the data for a given family and population name. Will overwrite the data if it already exists.
    """
    family = family if isinstance(family, Family) else lookup_family(family)
    data_dir = nn_data_path(family.key, population_name)

    if not data_dir.exists():
        family_root = nn_data_root.joinpath(family.key)

        if family_root.exists():
            print(f"No such population {population_name} for {family.key}", file=sys.stderr)
            print(f"Existing populations: ", file=sys.stderr)
            for population_dir in family_root.iterdir():
                print(f"\t '{population_dir.name}'", file=sys.stderr)
            raise FileNotFoundError(f"No such population {population_name} for {family.key}")
        else:
            print(f"No such family {family.key}", file=sys.stderr)
            print(f"Existing families: ", file=sys.stderr)
            for family_dir in nn_data_root.iterdir():
                print(f"\t '{family_dir.name}'", file=sys.stderr)
            raise FileNotFoundError(f"No data for family: '{family.key}'")

    x_data = jnp.load(data_dir.joinpath("x.npy"))
    y_data = jnp.load(data_dir.joinpath("y.npy"))

    verify_data_integrity(family, x_data, y_data)

    if filter_bad_data:
        filter_bad_trajectories(x_data, y_data, maximum_value)
    elif maximum_value:
        print("Warning: maximum_value is only used when filter_bad_data is True", file=sys.stderr)

    return x_data, y_data


def verify_data_integrity(family: Family, trajectories: jnp.ndarray, embeddings: jnp.ndarray):
    assert len(trajectories.shape) == 3, "Trajectories must have shape (samples, timesteps, 2 * dof)"
    assert len(embeddings.shape) == 2, "Embeddings must have shape (samples, embedding_size)"

    assert trajectories.shape[0] == embeddings.shape[0], \
        "Trajectories and embeddings must have the same number of samples"

    assert embeddings.shape[1] == family.embedding_shape[0], \
        "Embeddings must have the correct shape for the family"


def load_data_wrapped(
        family: Family | str,
        data_name: str,
        timestep_cap: int,
        datasize_cap: int | None = None,
        maximum_value: int = 10 ** 5
):
    # Load data
    x, y = load_nn_data(family, data_name)
    row_indices = jnp.where(jnp.any(x > maximum_value, axis=1))[0]
    x_mask = jnp.ones(x.shape[0], dtype=bool).at[row_indices].set(False)
    y_mask = jnp.ones(y.shape[0], dtype=bool).at[row_indices].set(False)
    x = x[x_mask]
    y = y[y_mask]
    x = x[:, :timestep_cap + 1, :]

    if datasize_cap is not None:
        x = x[:datasize_cap]
        y = y[:datasize_cap]

    return x, y

# def get_data(batch_size: int, dataName: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
#     # Reserve 10,000 samples for validation.
#     validation_cutoff = 10_000
#     x, y = load_data_wrapped(family, dataName, TRAINING_TIMESTEPS)
#
#     # Split into train and validation
#     x_val = x[-validation_cutoff:]
#     y_val = y[-validation_cutoff:]
#     x_train = x[:-validation_cutoff]
#     y_train = y[:-validation_cutoff]
#
#     if not (jnp.all(jnp.isfinite(x_train)) and jnp.all(jnp.isfinite(y_train))):
#         sys.exit('infs/NaNs in training data')
#     if not (jnp.all(jnp.isfinite(x_val)) and jnp.all(jnp.isfinite(y_val))):
#         sys.exit('infs/NaNs in validation data')
#
#     # Prepare the training dataset.
#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     train_dataset = train_dataset.shuffle(
#         buffer_size=1024,
#         seed=SHUFFLE_SEED,
#     ).batch(batch_size)
#
#     # Prepare the validation dataset.
#     val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#     val_dataset = val_dataset.batch(batch_size)
#
#     return train_dataset, val_dataset
