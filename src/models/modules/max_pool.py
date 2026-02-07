from flax import nnx

from utils.config.modules import MaxPoolConfig


def make_max_pool(
    cfg: MaxPoolConfig,
    **_,
):
    return lambda x: nnx.max_pool(x, window_shape=cfg.window_shape)
