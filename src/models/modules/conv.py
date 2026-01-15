from flax import nnx
from utils.config.modules import ConvConfig

from .acts import ACTS


def make_conv(cfg: ConvConfig, rngs: nnx.Rngs):
    return nnx.Sequential(
        nnx.Conv(
            in_features=cfg.in_features,
            out_features=cfg.out_features,
            kernel_size=cfg.kernel_size,
            strides=cfg.strides,
            padding=cfg.padding,
            rngs=rngs,
        ),
        nnx.BatchNorm(cfg.out_features, rngs=rngs),
        ACTS[cfg.act],
    )
