from pydantic import ValidationError
from torch.utils.data import DataLoader

from utils.config.datasets import DatasetConfig, PiVAEConfig

from .mnist import get_MNIST
from .pivae import get_pivae

DATASETS = {
    "mnist": get_MNIST,
    "pivae": get_pivae,
}

CONFIG = {
    "mnist": DatasetConfig,
    "pivae": PiVAEConfig,
}


def get_dataset(name: str, cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train and test dataloaders for the specified dataset and accompanying config"""
    if name in DATASETS and name in CONFIG:
        try:
            return DATASETS[name](CONFIG[name](**cfg))
        except ValidationError as e:
            print("Invalid config file!")
            for error in e.errors():
                print(f"  * {error['type'].capitalize()} field '{error['loc'][0]}'")
            quit()

    raise NotImplementedError(f"Dataset {name} not implemented")


__all__ = [
    "get_dataset",
]
