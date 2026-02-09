from os import PathLike

import yaml
from pydantic import ValidationError
from pydantic.dataclasses import dataclass


@dataclass
class ExperimentConfig:
    model: str
    model_config: dict
    dataset: str
    dataset_config: dict
    trainer_config: dict
    checkpoint_dir: str


@dataclass
class VisualiserConfig:
    folder: PathLike | str
    format: str = "png"
    checkpoint: str = "latest"
    n_rows: int = 5
    n_cols: int = 20
    seed: int = 0


def load_config(config_file: PathLike) -> ExperimentConfig:
    """
    Load and parse the provided config file.

    Parameters
    ----------
    config_file : PathLike
        The path to the config to load

    Returns
    -------
    config : ExperimentConfig
        The config for the given experiment
    """
    try:
        with open(config_file, "r") as file:
            return ExperimentConfig(**yaml.safe_load(file))
    except FileNotFoundError:
        print(f"Provided config file, {config_file} not found")
        quit()
    except yaml.YAMLError:
        print("Error processing the config file. Verify the YAML formatting")
        quit()
    except ValidationError as e:
        print("Failed to parse loaded config.")
        print(e.json(indent=2, include_url=False))
        quit()
