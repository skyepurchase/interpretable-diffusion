from flax import nnx


ACTS = {
    "relu": nnx.relu,
    "sigmoid": nnx.sigmoid,
    "identity": lambda x: x,
}
