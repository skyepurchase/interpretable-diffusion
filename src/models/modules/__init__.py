from flax import nnx

from utils.config.modules import ConvConfig, LinearConfig, MaxPoolConfig, UpSampleConfig

from .conv import make_conv
from .linear import make_linear
from .max_pool import make_max_pool
from .upsample import make_upsample

MODULES = {
    "conv": (make_conv, ConvConfig),
    "linear": (make_linear, LinearConfig),
    "maxpool": (make_max_pool, MaxPoolConfig),
    "upsample": (make_upsample, UpSampleConfig),
    "flatten": (lambda _, rngs: lambda x: x.reshape(x.shape[0], -1), lambda *_: None),
}

ACTS = {"swish": nnx.swish}


def make_module(type: str, cfg: dict, out_features: int, *, rngs: nnx.Rngs):
    if type in MODULES:
        module, make_config = MODULES[type]
        config = make_config(**cfg)

        if hasattr(config, "out_features") and config.out_features is None:
            config.out_features = out_features

        return module(config, rngs=rngs)


__all__ = ["make_module"]
