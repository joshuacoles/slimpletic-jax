from pathlib import Path

# The root directory where all data will be stored, $PROJECT_ROOT/data
project_data_root = Path(__file__).parent.parent.parent.joinpath('data')
nn_data_root = project_data_root.joinpath("nn_data")


def nn_data_path(family_key: str, population_name: str):
    return nn_data_root.joinpath(f"{family_key}/{population_name}")
