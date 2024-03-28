import jax.numpy as jnp

from neural_networks.data import nn_data_path
from neural_networks.data.families import Family


def load_data(family: Family, population_name: str):
    """
    Load the data for a given family and population name.
    """
    data_dir = nn_data_path(family.key, population_name)
    x_data = jnp.load(data_dir.joinpath("x.npy"))
    y_data = jnp.load(data_dir.joinpath("y.npy"))
    return x_data, y_data
