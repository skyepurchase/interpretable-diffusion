import jax.numpy as jnp
from jax.tree_util import tree_map

from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

import numpy as np
from numpy.typing import NDArray

from utils.config.datasets import DatasetConfig


def np_collate(batch: list) -> NDArray:
    """
    Carry out the default torch collation and convert to JAX array
    """
    return tree_map(np.asarray, default_collate(batch))


def jnp_cast(img) -> NDArray:
    """
    Cast the image to JAX array
    """
    data = np.array(img, dtype=jnp.float32)
    data /= data.max()

    return np.expand_dims(data, -1)


def get_MNIST(cfg: DatasetConfig) -> tuple[DataLoader, DataLoader]:
    train = MNIST(cfg.dir, download=True, transform=jnp_cast)
    test = MNIST(cfg.dir, download=True, transform=jnp_cast, train=False)

    train_loader = DataLoader(train, batch_size=cfg.batch_size, collate_fn=np_collate)
    test_loader = DataLoader(train, batch_size=len(test), collate_fn=np_collate)

    return train_loader, test_loader
