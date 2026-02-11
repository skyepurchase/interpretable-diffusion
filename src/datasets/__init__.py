from torch.utils.data import DataLoader

from utils.config.datasets import DatasetConfig

from .mnist import get_MNIST
from .pivae import get_pendulum

DATASETS = {
    "mnist": get_MNIST,
    "pendulum": get_pendulum,
}


def get_dataset(name: str, cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for the specified dataset and accompanying config"""
    if name in DATASETS:
        return DATASETS[name](DatasetConfig(**cfg))

    raise NotImplementedError(f"Dataset {name} not found")


__all__ = [
    "get_dataset",
]
