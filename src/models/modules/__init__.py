from .conv import make_conv
from .linear import make_linear
from .max_pool import make_max_pool
from .upsample import make_upsample

MODULES = {
    "conv": make_conv,
    "linear": make_linear,
    "maxpool": make_max_pool,
    "upsample": make_upsample,
    "flatten": lambda rngs: lambda x: x.reshape(x.shape[0], -1),
}
