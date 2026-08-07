import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from PIL.Image import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from datasets.utils import np_collate


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


def get_MNIST(
    dir: str, batch_size: int = 128, download: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create MNIST train, val and test dataloaders"""
    train_and_val = MNIST(dir, download=download, transform=normalise_cast)
    test = MNIST(dir, download=download, transform=normalise_cast, train=False)

    train_split = int(len(train_and_val) * 0.9)
    val_split = len(train_and_val) - train_split
    train, val = random_split(train_and_val, [train_split, val_split])

    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=np_collate)
    val_loader = DataLoader(
        val, batch_size=batch_size, collate_fn=np_collate, shuffle=False
    )
    test_loader = DataLoader(
        train_and_val, batch_size=len(test), collate_fn=np_collate, shuffle=False
    )

    return train_loader, val_loader, test_loader
