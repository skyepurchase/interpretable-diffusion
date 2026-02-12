import numpy as np
from jax.tree_util import tree_map
from numpy.typing import NDArray
from torch.utils.data import default_collate


def np_collate(batch: list) -> NDArray:
    """
    Carry out the default torch collation and convert to JAX array
    """
    return tree_map(np.asarray, default_collate(batch))
