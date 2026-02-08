from flax import nnx

from utils.config.modules import LinearConfig

from .acts import ACTS
from .utils import ModuleFn


def make_linear(cfg: LinearConfig, rngs: nnx.Rngs) -> ModuleFn:
    """Create an affine function"""
    assert cfg.out_features is not None

    return nnx.Sequential(
        nnx.Linear(cfg.in_features, cfg.out_features, rngs=rngs), ACTS[cfg.act]
    )
