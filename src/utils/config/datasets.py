from pydantic.dataclasses import dataclass


@dataclass
class DatasetConfig:
    dir: str
    download: bool = False
    batch_size: int = 128


@dataclass
class PiVAEConfig:
    dir: str
    system: str
    n_samples: int = 3500
    test_split: int = 500
    validation_split: int = 1000
    train_split: int = 1000
    timesteps: int = 50
    batch_size: int = 128


def get_full_dataset_name(dataset: str, config: dict) -> str:
    """
    Get the full hiphenated dataset name if the dataset has subnames

    Parameters
    ----------
    dataset: str
        The overarching dataset name
    config: dict
        The dataset config that contains subnames
    """
    if "system" in config:
        return "-".join([dataset, config["system"]])
    else:
        return dataset
