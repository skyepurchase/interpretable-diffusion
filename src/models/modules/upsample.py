import jax

from utils.config.modules import UpSampleConfig

from .utils import ModuleFn


def make_upsample(cfg: UpSampleConfig, **_) -> ModuleFn:
    """Create an upsample function"""
    return lambda x: jax.image.resize(
        x, shape=(x.shape[0], *cfg.shape, x.shape[3]), method=cfg.method
    )
