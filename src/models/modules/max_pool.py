from flax import nnx

from .utils import ModuleFn


def make_max_pool(
    rngs=None,
    **kwargs,
) -> ModuleFn:
    """Create a max pool function"""
    return lambda x: nnx.max_pool(x, **kwargs)
