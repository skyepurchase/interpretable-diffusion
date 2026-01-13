from flax import nnx

from utils.config.modules import LinearConfig
from .acts import ACTS


def make_linear(
    cfg: LinearConfig,
    rngs: nnx.Rngs
):
    assert cfg.out_features is not None

    return nnx.Sequential(
        nnx.Linear(cfg.in_features, cfg.out_features, rngs=rngs),
        ACTS[cfg.act]
    )
