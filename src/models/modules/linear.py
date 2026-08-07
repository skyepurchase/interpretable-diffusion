from flax import nnx

from .acts import ACTS
from .utils import ModuleFn


def make_linear(in_features, out_features, act, rngs: nnx.Rngs) -> ModuleFn:
    """Create an affine function"""
    assert out_features is not None

    return nnx.Sequential(nnx.Linear(in_features, out_features, rngs=rngs), ACTS[act])
