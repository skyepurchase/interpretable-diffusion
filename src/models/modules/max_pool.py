from flax import nnx

from utils.config.modules import MaxPoolConfig

from .utils import ModuleFn


def make_max_pool(
    cfg: MaxPoolConfig,
    **_,
) -> ModuleFn:
    """Create a max pool function"""
    return lambda x: nnx.max_pool(x, window_shape=cfg.window_shape, strides=cfg.strides)
