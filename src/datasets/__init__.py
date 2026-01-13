from utils.config.datasets import DatasetConfig

from .mnist import get_MNIST


DATASETS = {
    'mnist': get_MNIST
}


def get_dataset(name: str, cfg):
    if name in DATASETS:
        return DATASETS[name](DatasetConfig(**cfg))

    raise ValueError(f"Dataset {name} not found")


__all__ = [
    'get_dataset',
]
