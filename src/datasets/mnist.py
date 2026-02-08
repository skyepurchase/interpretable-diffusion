import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map
from numpy.typing import NDArray
from PIL.Image import Image
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

from utils.config.datasets import DatasetConfig


def np_collate(batch: list) -> NDArray:
    """
    Carry out the default torch collation and convert to JAX array
    """
    return tree_map(np.asarray, default_collate(batch))


def normalise_cast(img: Image) -> NDArray:
    """
    Normalise and cast an image
    """
    data = np.array(img, dtype=jnp.float32)
    data /= data.max()

    if len(data.shape) == 2:
        return np.expand_dims(data, -1)

    assert len(data.shape) == 3
    return data


def get_MNIST(cfg: DatasetConfig) -> tuple[DataLoader, DataLoader]:
    """Create MNIST train and test dataloaders"""
    train = MNIST(cfg.dir, download=True, transform=normalise_cast)
    test = MNIST(cfg.dir, download=True, transform=normalise_cast, train=False)

    train_loader = DataLoader(train, batch_size=cfg.batch_size, collate_fn=np_collate)
    test_loader = DataLoader(train, batch_size=len(test), collate_fn=np_collate)

    return train_loader, test_loader
