from flax import nnx

from utils.config.modules import (
    ConvConfig,
    LinearConfig,
    MaxPoolConfig,
    UpSampleConfig,
)
from .upsample import make_upsample
from .max_pool import make_max_pool
from .conv import make_conv
from .linear import make_linear


MODULES = {
    "conv": (make_conv, ConvConfig),
    "linear": (make_linear, LinearConfig),
    "maxpool": (make_max_pool, MaxPoolConfig),
    "upsample": (make_upsample, UpSampleConfig),
    "flatten": (lambda _, rngs: lambda x: x.reshape(x.shape[0], -1), lambda *_: None),
}


def make_module(
    name: str, cfg: dict, out_features: int, *, rngs: nnx.Rngs
) -> nnx.Module:
    if name in MODULES:
        module, make_config = MODULES[name]
        config = make_config(**cfg)

        if hasattr(config, "out_features") and config.out_features is None:
            config.out_features = out_features

        return module(config, rngs=rngs)

    raise KeyError(f"Module type {name} not found")


__all__ = ["make_module"]
