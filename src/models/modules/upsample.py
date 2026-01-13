import jax

from utils.config.modules import UpSampleConfig


def make_upsample(
    cfg: UpSampleConfig,
    **_
):
    return lambda x: jax.image.resize(
        x,
        shape=(x.shape[0], *cfg.shape, x.shape[3]),
        method=cfg.method
    )
