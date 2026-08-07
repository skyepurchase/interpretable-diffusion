import jax

from .utils import ModuleFn


def make_upsample(shape: tuple[int], method: str, **_) -> ModuleFn:
    """Create an upsample function"""
    return lambda x: jax.image.resize(
        x, shape=(x.shape[0], *shape, x.shape[3]), method=method
    )
