from flax import nnx

from models.modules.utils import ModuleFn
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


def make_module(name: str, cfg: dict, out_features: int, *, rngs: nnx.Rngs) -> ModuleFn:
    """
    Create the specified module according to the provided config.

    Parameters
    ----------
    name : str
        The name of the module
    cfg : dict
        The config that will be cast to the necessary dataclass
    out_features : int
        The target features of the model
    rngs : nnx.Rngs
        The run rng

    Returns
    -------
    module : ModuleFn
        The specified module
    """
    if name in MODULES:
        module, make_config = MODULES[name]
        config = make_config(**cfg)

        if hasattr(config, "out_features") and config.out_features is None:
            config.out_features = out_features

        return module(config, rngs=rngs)

    raise KeyError(f"Module type {name} not found")


__all__ = ["make_module"]
