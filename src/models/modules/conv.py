from flax import nnx

from .acts import ACTS
from .utils import ModuleFn


def make_conv(
    in_features, out_features, kernel_size, strides, padding, act, rngs: nnx.Rngs
) -> ModuleFn:
    """Create a convolutional block with batchnorm and activation"""
    return nnx.Sequential(
        nnx.Conv(
            in_features,
            out_features,
            kernel_size,
            strides,
            padding=padding,
            rngs=rngs,
        ),
        nnx.BatchNorm(out_features, rngs=rngs),
        ACTS[act],
    )
